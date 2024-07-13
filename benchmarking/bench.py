import itertools
import os
import subprocess
import copy
import json
import time
import traceback
from shutil import copy as shcopy
import finn.core.onnx_exec as oxe
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.transformation.fpgadataflow.make_zynq_proj import collect_ip_dirs
from finn.util.basic import make_build_dir, pynq_native_port_width, pynq_part_map
from dut.mvau import make_mvau_dut
from dut.transformer import make_transformer_dut
from templates import template_open, template_single_test, template_sim_power, template_switching_simulation_tb, zynq_harness_template
from util import summarize_table, summarize_section, power_xml_to_dict, prepare_inputs
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from qonnx.util.basic import (
    gen_finn_dt_tensor,
    roundup_to_integer_multiple,
)

class MakeZYNQHarnessProject(Transformation):
    """Based on MakeZYNQProject transformation, but integrates IP into test harness instead of DMA shell."""

    def __init__(self, platform, results_dir, save_dir, run_id, dut_duplication=1, clock_period_ns=10):
        super().__init__()
        self.platform = platform
        self.results_dir = results_dir
        self.save_dir = save_dir
        self.run_id = run_id
        self.dut_duplication = dut_duplication
        self.clock_period_ns = clock_period_ns

    def apply(self, model):
        # create a config file and empty list of xo files
        config = []
        idma_idx = 0
        odma_idx = 0
        aximm_idx = 0
        axilite_idx = 0
        global_clk_ns = 0

        # assume single stitched-ip (previously dataflowpartition) as DUT
        node = model.graph.node[0]
        node_inst = getCustomOp(node)
        instream_width = node_inst.get_instream_width_padded()
        outstream_width = node_inst.get_outstream_width_padded()
        # TODO: make compatible with multi-layer DUTs

        # assert node.op_type == "StreamingDataflowPartition", "Invalid link graph"
        # sdp_node = getCustomOp(node)
        # dataflow_model_filename = sdp_node.get_nodeattr("model")
        # kernel_model = ModelWrapper(dataflow_model_filename)
        kernel_model = model

        ipstitch_path = kernel_model.get_metadata_prop("vivado_stitch_proj")
        if ipstitch_path is None or (not os.path.isdir(ipstitch_path)):
            raise Exception(
                "No stitched IPI design found for %s, apply CreateStitchedIP first." % node.name
            )

        vivado_stitch_vlnv = kernel_model.get_metadata_prop("vivado_stitch_vlnv")
        if vivado_stitch_vlnv is None:
            raise Exception("No vlnv found for %s, apply CreateStitchedIP first." % node.name)

        ip_dirs = ["list"]
        ip_dirs += collect_ip_dirs(kernel_model, ipstitch_path)
        ip_dirs.append("$::env(FINN_ROOT)/benchmarking/harness/sink/ip")
        ip_dirs_str = "[%s]" % (" ".join(ip_dirs))
        config.append(
            "set_property ip_repo_paths "
            "[concat [get_property ip_repo_paths [current_project]] %s] "
            "[current_project]" % ip_dirs_str
        )
        config.append("update_ip_catalog -rebuild -scan_changes")
        config.append(
            "import_files -fileset sources_1 -norecurse $::env(FINN_ROOT)/benchmarking/harness/vector_xor.v"
        )

        # get metadata property clk_ns to calculate clock frequency
        clk_ns = float(kernel_model.get_metadata_prop("clk_ns"))
        if clk_ns > global_clk_ns:
            global_clk_ns = clk_ns

        ifnames = eval(kernel_model.get_metadata_prop("vivado_stitch_ifnames"))

        # instantiate DUT, TODO: switch to wrapper verilog file for (multiple-) DUT instantiation
        for id in range(self.dut_duplication):
            dut_instance_name = "finn_design_%d" % id
            config.append(
                "create_bd_cell -type ip -vlnv %s %s" % (vivado_stitch_vlnv, dut_instance_name)
            )
            # sdp_node.set_nodeattr("instance_name", instance_names[node.name])
            config.append(
                "connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins axi_interconnect_0/aclk]"
                % dut_instance_name
            )
            config.append(
                "connect_bd_net [get_bd_pins %s/ap_rst_n] [get_bd_pins axi_interconnect_0/aresetn]"
                % dut_instance_name
            )

        # instantiate input harness
        if instream_width > 8192:
            print("ERROR: DUT input stream width > 8192")
            raise Exception("ERROR: DUT input stream width > 8192")
        elif instream_width > 4096:
            num_sources = 8
            source_width = roundup_to_integer_multiple(instream_width / 8, 8)
        elif instream_width > 2048:
            num_sources = 4
            source_width = roundup_to_integer_multiple(instream_width / 4, 8)
        elif instream_width > 1024:
            num_sources = 2
            source_width = roundup_to_integer_multiple(instream_width / 2, 8)
        else:
            num_sources = 1
            source_width = instream_width

        if self.dut_duplication > 1:
            if num_sources > 1:
                print("ERROR: DUT duplication with >1024 stream width not supported!")
                raise Exception("ERROR: DUT duplication with >1024 stream width not supported!")

            num_sources = self.dut_duplication  # one source per DUT instance
            seed = 0xABCD
            for id in range(num_sources):
                config.append(
                    "create_bd_cell -type ip -vlnv xilinx.com:ip:axi_traffic_gen:3.0 axi_traffic_gen_%d"
                    % id
                )
                config.append(
                    "set_property -dict [list \
                    CONFIG.C_ATG_MODE {AXI4-Stream} \
                    CONFIG.C_ATG_STREAMING_MAX_LEN_BITS {1} \
                    CONFIG.C_AXIS_SPARSE_EN {false} \
                    CONFIG.C_AXIS_TDATA_WIDTH {%d} \
                    CONFIG.C_AXIS_TDEST_WIDTH {0} \
                    CONFIG.C_AXIS_TID_WIDTH {0} \
                    CONFIG.C_AXIS_TUSER_WIDTH {0} \
                    CONFIG.STRM_DATA_SEED {%s} \
                    ] [get_bd_cells axi_traffic_gen_%d]"
                    % (source_width, "0x{:04X}".format(seed), id)
                )
                config.append(
                    "connect_bd_net [get_bd_pins axi_traffic_gen_%d/s_axi_aclk] [get_bd_pins axi_interconnect_0/aclk]"
                    % id
                )
                config.append(
                    "connect_bd_net [get_bd_pins axi_traffic_gen_%d/s_axi_aresetn] [get_bd_pins axi_interconnect_0/aresetn]"
                    % id
                )
                seed = seed + 99

                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins axi_traffic_gen_%d/M_AXIS_MASTER] [get_bd_intf_pins finn_design_%d/s_axis_0]"
                    % (id, id)
                )

        else:
            seed = 0xABCD
            for id in range(num_sources):
                config.append(
                    "create_bd_cell -type ip -vlnv xilinx.com:ip:axi_traffic_gen:3.0 axi_traffic_gen_%d"
                    % id
                )
                config.append(
                    "set_property -dict [list \
                    CONFIG.C_ATG_MODE {AXI4-Stream} \
                    CONFIG.C_ATG_STREAMING_MAX_LEN_BITS {1} \
                    CONFIG.C_AXIS_SPARSE_EN {false} \
                    CONFIG.C_AXIS_TDATA_WIDTH {%d} \
                    CONFIG.C_AXIS_TDEST_WIDTH {0} \
                    CONFIG.C_AXIS_TID_WIDTH {0} \
                    CONFIG.C_AXIS_TUSER_WIDTH {0} \
                    CONFIG.STRM_DATA_SEED {%s} \
                    ] [get_bd_cells axi_traffic_gen_%d]"
                    % (source_width, "0x{:04X}".format(seed), id)
                )
                config.append(
                    "connect_bd_net [get_bd_pins axi_traffic_gen_%d/s_axi_aclk] [get_bd_pins axi_interconnect_0/aclk]"
                    % id
                )
                config.append(
                    "connect_bd_net [get_bd_pins axi_traffic_gen_%d/s_axi_aresetn] [get_bd_pins axi_interconnect_0/aresetn]"
                    % id
                )
                config.append(
                    "connect_bd_net [get_bd_pins finn_design_0/s_axis_0_tready] [get_bd_pins axi_traffic_gen_%d/m_axis_1_tready]"
                    % id
                )
                seed = seed + 99

            if num_sources > 1:
                config.append(
                    "create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_tdata"
                )
                config.append(
                    "set_property CONFIG.NUM_PORTS {%d} [get_bd_cells xlconcat_tdata]" % num_sources
                )

                for id in range(num_sources):
                    config.append(
                        "connect_bd_net [get_bd_pins xlconcat_tdata/In%d] [get_bd_pins axi_traffic_gen_%d/m_axis_1_tdata]"
                        % (id, id)
                    )

                config.append(
                    "connect_bd_net [get_bd_pins finn_design_0/s_axis_0_tdata] [get_bd_pins xlconcat_tdata/dout]"
                )
            else:
                config.append(
                    "connect_bd_net [get_bd_pins finn_design_0/s_axis_0_tdata] [get_bd_pins axi_traffic_gen_0/m_axis_1_tdata]"
                )

            # only connect valid from source 0 to DUT
            config.append(
                "connect_bd_net [get_bd_pins finn_design_0/s_axis_0_tvalid] [get_bd_pins axi_traffic_gen_0/m_axis_1_tvalid]"
            )

        # instantiate output harness
        for id in range(self.dut_duplication):
            config.append(
                "create_bd_cell -type ip -vlnv xilinx.com:user:harness_sink:1.0 sink_%d" % id
            )
            config.append(
                "set_property -dict [list CONFIG.STREAM_WIDTH {%d}] [get_bd_cells sink_%d]"
                % (outstream_width, id)
            )
            config.append(
                "connect_bd_intf_net [get_bd_intf_pins sink_%d/s_axis_0] [get_bd_intf_pins finn_design_%d/m_axis_0]"
                % (id, id)
            )

        # GPIO control (TODO: connect interrupt)
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0")
        config.append(
            "set_property -dict [list \
            CONFIG.C_ALL_INPUTS {0} \
            CONFIG.C_GPIO_WIDTH {5} \
            CONFIG.C_INTERRUPT_PRESENT {1} \
            ] [get_bd_cells axi_gpio_0]"
        )
        config.append(
            "connect_bd_intf_net [get_bd_intf_pins axi_gpio_0/S_AXI] "
            "[get_bd_intf_pins axi_interconnect_0/M%02d_AXI]" % (axilite_idx)
        )
        config.append("assign_axi_addr_proc axi_gpio_0/S_AXI")
        axilite_idx += 1
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_0")
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_1")
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_2")
        config.append(
            "set_property -dict [list \
            CONFIG.DIN_FROM {0} \
            CONFIG.DIN_TO {0} \
            CONFIG.DIN_WIDTH {5} \
            ] [get_bd_cells xlslice_0]"
        )
        config.append(
            "set_property -dict [list \
            CONFIG.DIN_FROM {1} \
            CONFIG.DIN_TO {1} \
            CONFIG.DIN_WIDTH {5} \
            ] [get_bd_cells xlslice_1]"
        )
        config.append(
            "set_property -dict [list \
            CONFIG.DIN_FROM {2} \
            CONFIG.DIN_TO {2} \
            CONFIG.DIN_WIDTH {5} \
            ] [get_bd_cells xlslice_2]"
        )
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0")
        config.append(
            "set_property -dict [list CONFIG.IN1_WIDTH.VALUE_SRC USER CONFIG.IN2_WIDTH.VALUE_SRC USER CONFIG.IN0_WIDTH.VALUE_SRC USER] [get_bd_cells xlconcat_0]"
        )
        config.append(
            "set_property -dict [list \
            CONFIG.IN0_WIDTH {3} \
            CONFIG.NUM_PORTS {3} \
            ] [get_bd_cells xlconcat_0]"
        )
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0")
        config.append(
            "set_property -dict [list \
            CONFIG.CONST_VAL {0} \
            CONFIG.CONST_WIDTH {3} \
            ] [get_bd_cells xlconstant_0]"
        )
        config.append(
            """
            connect_bd_net [get_bd_pins xlslice_0/Din] [get_bd_pins axi_gpio_0/gpio_io_o]
            connect_bd_net [get_bd_pins xlslice_1/Din] [get_bd_pins axi_gpio_0/gpio_io_o]
            connect_bd_net [get_bd_pins xlslice_2/Din] [get_bd_pins axi_gpio_0/gpio_io_o]
            connect_bd_net [get_bd_pins xlconstant_0/dout] [get_bd_pins xlconcat_0/In0]
            connect_bd_net [get_bd_pins axi_gpio_0/gpio_io_i] [get_bd_pins xlconcat_0/dout]
        """
        )
        if self.dut_duplication > 1:
            config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_valid")
            config.append(
                "set_property CONFIG.NUM_PORTS {%d} [get_bd_cells xlconcat_valid]"
                % self.dut_duplication
            )
            config.append(
                "create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_checksum"
            )
            config.append(
                "set_property CONFIG.NUM_PORTS {%d} [get_bd_cells xlconcat_checksum]"
                % self.dut_duplication
            )

            config.append("create_bd_cell -type module -reference vector_xor vector_xor_valid")
            config.append(
                "set_property CONFIG.WIDTH {%d} [get_bd_cells vector_xor_valid]"
                % self.dut_duplication
            )
            config.append("create_bd_cell -type module -reference vector_xor vector_xor_checksum")
            config.append(
                "set_property CONFIG.WIDTH {%d} [get_bd_cells vector_xor_checksum]"
                % self.dut_duplication
            )

            config.append(
                "connect_bd_net [get_bd_pins vector_xor_valid/in_data] [get_bd_pins xlconcat_valid/dout]"
            )
            config.append(
                "connect_bd_net [get_bd_pins vector_xor_checksum/in_data] [get_bd_pins xlconcat_checksum/dout]"
            )
            config.append(
                "connect_bd_net [get_bd_pins vector_xor_valid/out_data] [get_bd_pins xlconcat_0/In1]"
            )
            config.append(
                "connect_bd_net [get_bd_pins vector_xor_checksum/out_data] [get_bd_pins xlconcat_0/In2]"
            )
            for id in range(self.dut_duplication):
                config.append(
                    "connect_bd_net [get_bd_pins sink_%d/valid] [get_bd_pins xlconcat_valid/In%d]"
                    % (id, id)
                )
                config.append(
                    "connect_bd_net [get_bd_pins sink_%d/checksum] [get_bd_pins xlconcat_checksum/In%d]"
                    % (id, id)
                )
        else:
            config.append("connect_bd_net [get_bd_pins sink_0/valid] [get_bd_pins xlconcat_0/In1]")
            config.append(
                "connect_bd_net [get_bd_pins sink_0/checksum] [get_bd_pins xlconcat_0/In2]"
            )
        for id in range(self.dut_duplication):
            config.append(
                "connect_bd_net [get_bd_pins xlslice_2/Dout] [get_bd_pins sink_%d/enable]" % id
            )
        for id in range(num_sources):
            config.append(
                "connect_bd_net [get_bd_pins xlslice_0/Dout] [get_bd_pins axi_traffic_gen_%d/core_ext_start]"
                % id
            )
            config.append(
                "connect_bd_net [get_bd_pins xlslice_1/Dout] [get_bd_pins axi_traffic_gen_%d/core_ext_stop]"
                % id
            )

        # create a temporary folder for the project
        vivado_pynq_proj_dir = make_build_dir(prefix="vivado_zynq_proj_")
        model.set_metadata_prop("vivado_pynq_proj", vivado_pynq_proj_dir)

        fclk_mhz = int(1 / (global_clk_ns * 0.001))

        # create a TCL recipe for the project
        ipcfg = vivado_pynq_proj_dir + "/ip_config.tcl"
        config = "\n".join(config) + "\n"
        with open(ipcfg, "w") as f:
            f.write(
                zynq_harness_template
                % (
                    fclk_mhz,
                    axilite_idx,
                    aximm_idx,
                    self.platform,
                    pynq_part_map[self.platform],
                    config,
                )
            )

        # create a TCL recipe for the project
        synth_project_sh = vivado_pynq_proj_dir + "/synth_project.sh"
        working_dir = os.environ["PWD"]
        with open(synth_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(vivado_pynq_proj_dir))
            f.write("vivado -mode batch -source %s\n" % ipcfg)
            f.write("cd {}\n".format(working_dir))

        # call the synthesis script
        bash_command = ["bash", synth_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()

        # collect results
        results_dir_bitstreams = os.path.join(self.results_dir, "bitstreams")
        os.makedirs(results_dir_bitstreams, exist_ok=True)

        bitfile_name = vivado_pynq_proj_dir + "/finn_zynq_link.runs/impl_1/top_wrapper.bit"
        if not os.path.isfile(bitfile_name):
            raise Exception(
                "Synthesis failed, no bitfile found. Check logs under %s" % vivado_pynq_proj_dir
            )
        deploy_bitfile_name = results_dir_bitstreams + "/run_%d.bit" % self.run_id
        shcopy(bitfile_name, deploy_bitfile_name)

        hwh_name = vivado_pynq_proj_dir + "/finn_zynq_link.gen/sources_1/bd/top/hw_handoff/top.hwh"
        if not os.path.isfile(hwh_name):
            raise Exception(
                "Synthesis failed, no hwh file found. Check logs under %s" % vivado_pynq_proj_dir
            )
        deploy_hwh_name = results_dir_bitstreams + "/run_%d.hwh" % self.run_id
        shcopy(hwh_name, deploy_hwh_name)

        synth_report_filename = vivado_pynq_proj_dir + "/synth_report.xml"
        deploy_synth_report_filename = results_dir_bitstreams + "/run_%d.xml" % self.run_id
        shcopy(synth_report_filename, deploy_synth_report_filename)

        # copy additionally to a separate PFS location for measurement automation to pick it up
        # TODO: make this more configurable or switch to job/artifact based power measurement 
        shcopy(deploy_bitfile_name, self.save_dir)
        shcopy(deploy_hwh_name, self.save_dir)
        shcopy(deploy_synth_report_filename, self.save_dir)

        clock_period_mhz = int(1.0 / self.clock_period_ns * 1000.0)
        measurement_settings = {"freq_mhz": clock_period_mhz}
        with open(os.path.join(self.save_dir, "run_%d_settings.json"%self.run_id), "w") as f:
            json.dump(measurement_settings, f, indent=2)

        # model.set_metadata_prop("bitfile", deploy_bitfile_name)
        # model.set_metadata_prop("hw_handoff", deploy_hwh_name)
        # model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)
        return (model, False)

def start_test_batch_fast(results_path, project_path, run_target, pairs):
    # Prepare tcl script
    script = template_open.replace("$PROJ_PATH$", project_path)
    # script = script.replace("$PERIOD$", period)
    script = script.replace("$RUN$", run_target)
    for toggle_rate, static_prob in pairs:
        script = script + template_single_test
        script = script.replace("$TOGGLE_RATE$", str(toggle_rate))
        script = script.replace("$STATIC_PROB$", str(static_prob))
        # script = script.replace("$SWITCH_TARGET$", switch_target)
        script = script.replace("$REPORT_PATH$", results_path)
        script = script.replace("$REPORT_NAME$", f"{toggle_rate}_{static_prob}")
    with open(os.getcwd() + "/power_report.tcl", "w") as tcl_file:
        tcl_file.write(script)

    # Prepare bash script
    bash_script = os.getcwd() + "/report_power.sh"
    with open(bash_script, "w") as script:
        script.write("#!/bin/bash \n")
        script.write(f"vivado -mode batch -source {os.getcwd()}/power_report.tcl\n")

    # Run script
    sub_proc = subprocess.Popen(["bash", bash_script])
    sub_proc.communicate()

    # Parse results
    for toggle_rate, static_prob in pairs:
        power_report_dict = power_xml_to_dict(f"{results_path}/{toggle_rate}_{static_prob}.xml")
        power_report_json = f"{results_path}/{toggle_rate}_{static_prob}.json"
        with open(power_report_json, "w") as json_file:
            json_file.write(json.dumps(power_report_dict, indent=2))


def sim_power_report(results_path, project_path, in_width, out_width, dtype_width, sim_duration_ns):
    # Prepare tcl script
    script = template_open.replace("$PROJ_PATH$", project_path)
    script = script.replace("$RUN$", "impl_1")
    script = script + template_sim_power
    script = script.replace("$TB_FILE_PATH$", os.getcwd() + "/switching_simulation_tb.v")
    script = script.replace("$SAIF_FILE_PATH$", os.getcwd() + "/switching.saif")
    script = script.replace("$SIM_DURATION_NS$", str(int(sim_duration_ns)))
    script = script.replace("$REPORT_PATH$", results_path)
    script = script.replace("$REPORT_NAME$", f"sim")
    with open(os.getcwd() + "/power_report.tcl", "w") as tcl_file:
        tcl_file.write(script)

    # Prepare testbench
    testbench = template_switching_simulation_tb.replace("$INSTREAM_WIDTH$", str(in_width))
    testbench = testbench.replace("$OUTSTREAM_WIDTH$", str(out_width))
    testbench = testbench.replace("$DTYPE_WIDTH$", str(dtype_width))
    testbench = testbench.replace(
        "$RANDOM_FUNCTION$", "$urandom_range(0, {max})".format(max=2**dtype_width - 1)
    )
    with open(os.getcwd() + "/switching_simulation_tb.v", "w") as tb_file:
        tb_file.write(testbench)

    # Prepare shell script
    bash_script = os.getcwd() + "/report_power.sh"
    with open(bash_script, "w") as script:
        script.write("#!/bin/bash \n")
        script.write(f"vivado -mode batch -source {os.getcwd()}/power_report.tcl\n")

    # Run script
    sub_proc = subprocess.Popen(["bash", bash_script])
    sub_proc.communicate()

    # Parse results
    power_report_dict = power_xml_to_dict(f"{results_path}/sim.xml")
    power_report_json = f"{results_path}/sim.json"
    with open(power_report_json, "w") as json_file:
        json_file.write(json.dumps(power_report_dict, indent=2))

class bench():
    def __init__(self, dut, params, task_id, run_id, results_dir, save_dir):
        super().__init__()
        self.dut = dut
        self.params = params
        self.task_id = task_id
        self.run_id = run_id
        self.results_dir = results_dir
        self.save_dir = save_dir

        # General configuration
        self.board = "RFSoC2x2"
        self.part = "xczu28dr-ffvg1517-2-e"  # TODO: make configurable, + Alveo support?
        self.clock_period_ns = 5

        # Initialize output directories (might exist from other runs of the same job)
        self.results_dir_models = os.path.join(self.results_dir, "models")
        os.makedirs(self.results_dir_models, exist_ok=True)
        self.results_dir_power = os.path.join(self.results_dir, "power", "run_%d" % (self.run_id))
        os.makedirs(self.results_dir_power, exist_ok=True)

        # Intermediate models saved between steps
        self.model_initial = None
        self.model_step_hls = None
        self.model_step_synthesis = None

        # Initialize dictionary to collect all benchmark results
        self.output_dict = {}
    
    def step_make_dut(self):
        # Make DUT TODO: implement these DUT-specific functions in respective subclasses
        if self.dut == "mvau":
            model, dut_info = make_mvau_dut(self.params)
            self.target_node = "MVAU_hls" # display results of analysis passes only for the first occurence of this op type
        elif self.dut == "transformer":
            model, dut_info = make_transformer_dut(self.params)
            self.target_node = None # aggregate results of analysis passes over all nodes in the DUT
        else:
            print("ERROR: unknown DUT specified")

        # Save model for logging purposes
        # TODO: benchmarking infrastructure could be integrated deeper into ONNX IR and FINN custom_op/transformation infrastructure
        # E.g. parameters and paths could be stored as onnx attributes and benchmarking steps as generic or specialized custom_op transformations
        model.save(os.path.join(self.results_dir_models, "model_%d_initial.onnx" % (self.run_id)))

        # Save model for use in other steps
        self.model_initial = model

        # Log dict reported by DUT-specific scripts to overall result dict
        # E.g. this could contain SIMD/PE derived from folding factors or weight distribution information
        self.output_dict["info"] = dut_info

    def step_finn_estimate(self):
        # Gather FINN estimates
        print("Gathering FINN estimates")

        model = self.model_initial
        finn_resources_model = res_estimation(model, fpgapart=self.part)
        finn_cycles_model = model.analysis(exp_cycles_per_layer)
        if self.target_node:
            node = model.get_nodes_by_op_type(self.target_node)[0]
            finn_resources = finn_resources_model[node.name]
            finn_cycles = finn_cycles_model[node.name]
        else:
            finn_resources = finn_resources_model # TODO: aggregate?
            finn_cycles = 0 # TODO: aggregate or drop
        finn_estimates = finn_resources
        finn_estimates["CYCLES"] = finn_cycles
        self.output_dict["finn_estimates"] = finn_estimates

    def step_hls(self):
        # Perform Vitis HLS synthesis for HLS resource/performance reports
        start_time = time.time()
        print("Performing Vitis HLS synthesis")
        model = self.model_initial
        model = model.transform(PrepareIP(self.part, self.clock_period_ns))
        model = model.transform(HLSSynthIP())

        hls_resources_model = model.analysis(hls_synth_res_estimation)
        if self.target_node:
            node = model.get_nodes_by_op_type(self.target_node)[0]
            hls_resources = hls_resources_model[node.name]
        else:
            hls_resources = hls_resources_model # TODO: aggregate?
        self.output_dict["hls_estimates"] = hls_resources
        self.output_dict["hls_time"] = int(time.time() - start_time)

        self.model_step_hls = copy.deepcopy(model)

    def step_rtlsim(self):
        # Perform RTL simulation for performance measurement
        start_time = time.time()
        print("Performing Verilator RTL simulation (n=1)")
        # Prepare
        model = self.model_step_hls
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
        # Generate input data
        input_tensor = model.graph.input[0]
        input_shape = model.get_tensor_shape(input_tensor)
        input_dtype = model.get_tensor_datatype(input_tensor)
        x = gen_finn_dt_tensor(input_dtype, input_shape)
        input_dict = prepare_inputs(x, input_dtype, None) # TODO: fix Bipolar conversion case
        # Run
        oxe.execute_onnx(model, input_dict)["outp"]  # do not check output for correctness TODO: add functional verification throughout benchmarking steps
        # Log result
        node = model.get_nodes_by_op_type("MVAU_hls")[0]
        inst = getCustomOp(node)
        rtlsim_cycles = inst.get_nodeattr("cycles_rtlsim")
        self.output_dict["rtlsim_cycles"] = rtlsim_cycles
        self.output_dict["rtlsim_time"] = int(time.time() - start_time)

    def step_synthesis(self):
        # Perform Vivado synthesis for accurate resource/timing and inaccurate power reports

        start_time = time.time()
        print("Performing Vivado (stitched-ip, out-of-context) synthesis")
        model = self.model_step_hls
        model = model.transform(ReplaceVerilogRelPaths())
        model = model.transform(CreateStitchedIP(self.part, self.clock_period_ns))
        model = model.transform(SynthOutOfContext(part=self.part, clk_period_ns=self.clock_period_ns))
        ooc_synth_results = eval(model.get_metadata_prop("res_total_ooc_synth"))

        start_test_batch_fast(
            results_path=self.results_dir_power,
            project_path=os.path.join(
                ooc_synth_results["vivado_proj_folder"], "vivadocompile", "vivadocompile.xpr"
            ),
            run_target="impl_1",
            pairs=[(25, 0.5), (50, 0.5), (75, 0.5)],
        )

        # Log most important power results directly (refer to detailed logs for more)
        for reportname in ["25_0.5", "50_0.5", "75_0.5"]:
            with open(os.path.join(self.results_dir_power, "%s.json" % reportname), "r") as f:
                report = json.load(f)
                power = float(report["Summary"]["tables"][0]["Total On-Chip Power (W)"][0])
                power_dyn = float(report["Summary"]["tables"][0]["Dynamic (W)"][0])
                ooc_synth_results["power_%s" % reportname] = power
                ooc_synth_results["power_dyn_%s" % reportname] = power_dyn

        self.output_dict["ooc_synth"] = ooc_synth_results
        self.output_dict["ooc_synth_time"] = int(time.time() - start_time)

        # Save model for logging purposes
        model.save(os.path.join(self.results_dir_models, "model_%d_synthesis.onnx" % (self.run_id)))
        self.model_step_synthesis = copy.deepcopy(model)

    def step_sim_power(self):
        # Perform Vivado simulation for accurate power report
        start_time = time.time()
        if "ooc_synth" not in self.output_dict:
            print("ERROR: step_sim_power requires step_synthesis")
        print("Performing Vivado simulation for power report")
        if "rtlsim_cycles" in self.output_dict:
            sim_duration_ns = self.output_dict["rtlsim_cycles"] * 3 * self.clock_period_ns
        else:
            sim_duration_ns = self.output_dict["finn_estimates"]["CYCLES"] * 3 * self.clock_period_ns

        model = self.model_step_synthesis
        input_tensor = model.graph.input[0]
        output_tensor = model.graph.output[0]
        input_node_inst = getCustomOp(model.find_consumer(input_tensor))
        output_node_inst = getCustomOp(model.find_producer(output_tensor))
        sim_power_report(
            results_path=self.results_dir_power,
            project_path=os.path.join(
                self.output_dict["ooc_synth"]["vivado_proj_folder"], "vivadocompile", "vivadocompile.xpr"
            ),
            in_width=input_node_inst.get_instream_width(),
            out_width=output_node_inst.get_outstream_width(),
            dtype_width=model.get_tensor_datatype(input_tensor).bitwidth(), #TODO: check if this really works
            sim_duration_ns=sim_duration_ns,
        )

        # Log most important power results directly (refer to detailed logs for more)
        for reportname in ["sim"]:
            with open(os.path.join(self.results_dir_power, "%s.json" % reportname), "r") as f:
                report = json.load(f)
                power = float(report["Summary"]["tables"][0]["Total On-Chip Power (W)"][0])
                power_dyn = float(report["Summary"]["tables"][0]["Dynamic (W)"][0])
                self.output_dict["power_%s" % reportname] = power
                self.output_dict["power_dyn%s" % reportname] = power_dyn

        self.output_dict["sim_power_time"] = int(time.time() - start_time)

    def step_synth_power(self):
        # Perform Vivado synthesis for on-hardware power measurement
        start_time = time.time()
        if self.model_step_hls is None:
            print("ERROR: step_synth_power requires step_hls")
        print("Performing Vivado synthesis with test harness integration for power measurement")

        if "dut_duplication" in self.params:
            dut_duplication = self.params["dut_duplication"]
        else:
            dut_duplication = 1
    
        model = self.model_step_hls.transform(ReplaceVerilogRelPaths())
        model = model.transform(CreateStitchedIP(self.part, self.clock_period_ns))
        model = model.transform(
            MakeZYNQHarnessProject(
                platform=self.board,
                results_dir=self.results_dir,
                save_dir=self.save_dir,
                run_id=self.run_id,
                dut_duplication=dut_duplication,
                clock_period_ns=self.clock_period_ns
            )
        )

        self.output_dict["synth_power_time"] = int(time.time() - start_time)

        # Save model for logging purposes
        model.save(os.path.join(self.results_dir_models, "model_%d_synth_power.onnx" % (self.run_id)))

    def run(self):
        do_hls = self.params["do_hls"]
        do_rtlsim = self.params["do_rtlsim"]
        do_synthesis = self.params["do_synthesis"]
        do_sim_power = self.params["do_sim_power"]
        do_synth_power = self.params["do_synth_power"]

        # Perform steps
        self.step_make_dut()
        self.step_finn_estimate()

        if do_hls:
            self.step_hls()
        if do_rtlsim:
            self.step_rtlsim()
        if do_synthesis:
            self.step_synthesis()
        if do_sim_power:
            self.step_sim_power()
        if do_synth_power:
            self.step_synth_power()

        return self.output_dict


def main():
    # Gather job array info
    job_id = int(os.environ["SLURM_JOB_ID"])
    print("Job launched with ID: %d" % (job_id))
    try:
        array_id = int(os.environ["SLURM_ARRAY_JOB_ID"])
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        print(
            "Launched as job array (Array ID: %d, Task ID: %d, Task count: %d)"
            % (array_id, task_id, task_count)
        )
    except KeyError:
        array_id = job_id
        task_id = 0
        task_count = 1
        print("Launched as single job")

    # Prepare result directory
    # experiment_dir = os.environ.get("EXPERIMENT_DIR") # original experiment dir (before potential copy to ramdisk)
    experiment_dir = os.environ.get("CI_PROJECT_DIR")

    results_dir = os.path.join(experiment_dir, "bench_results")
    print("Collecting results in path: %s" % results_dir)
    os.makedirs(os.path.join(results_dir, "results"), exist_ok=True)
    log_path = os.path.join(results_dir, "results", "task_%d.json" % (task_id))
    
    # save dir for saving bitstreams (and optionally full build artifacts for debugging (TODO))
    # TODO: make this more configurable or switch to job/artifact based power measurement
    save_dir = os.path.join("/scratch/hpc-prf-radioml/felix/jobs/",
                            "CI_" + os.environ.get("CI_PIPELINE_IID") + "_" + os.environ.get("CI_PIPELINE_NAME"),
                            "bench_results", "bitstreams")
    print("Saving additional artifacts in path: %s" % save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Gather benchmarking configs
    # TODO: allow to instead accept a single json file as pipeline variable
    configs_path = os.path.join(os.path.dirname(__file__), "cfg")
    for config in os.listdir(configs_path):
        # only evaluate first config file found
        # TODO: allow arbitrary number of dut configs (maybe via spawning separate pipelines/jobs?)
        config_select = config
        break
   
    # Determine which DUT to run
    if config_select.startswith("mvau"):
        dut = "mvau"
    elif config_select.startswith("transformer"):
        dut = "transformer"
    else:
        print("ERROR: unknown DUT specified")
    print("Running benchmark for design-under-test %s" % (dut))

    # Load config (given relative to this script)
    config_path = os.path.join(configs_path, config_select)
    print("Loading config %s" % (config_path))
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        print("ERROR: config file not found")
        return

    # Expand all specified config combinations (gridsearch)
    config_expanded = []
    for param_set in config:
        param_set_expanded = list(
            dict(zip(param_set.keys(), x)) for x in itertools.product(*param_set.values())
        )
        config_expanded.extend(param_set_expanded)

    # Save config (only first job of array) for logging purposes
    if task_id == 0:
        with open(os.path.join(results_dir, "bench_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        with open(os.path.join(results_dir, "bench_config_exp.json"), "w") as f:
            json.dump(config_expanded, f, indent=2)

    # Determine which runs this job will work on
    total_runs = len(config_expanded)
    if total_runs <= task_count:
        if task_id < total_runs:
            selected_runs = [task_id]
        else:
            return
    else:
        selected_runs = []
        idx = task_id
        while idx < total_runs:
            selected_runs.append(idx)
            idx = idx + task_count
    print("This job will perform %d out of %d total runs" % (len(selected_runs), total_runs))

    # Run benchmark
    log = []
    for run, run_id in enumerate(selected_runs):
        print(
            "Starting run %d/%d (id %d of %d total runs)"
            % (run + 1, len(selected_runs), run_id, total_runs)
        )

        params = config_expanded[run_id]
        print("Run parameters: %s" % (str(params)))

        log_dict = {"run_id": run_id, "task_id": task_id, "params": params}

        start_time = time.time()
        try:
            bench_object = bench(dut, params, task_id, run_id, results_dir, save_dir)
            output_dict = bench_object.run()
            if output_dict is None:
                output_dict = {}
                log_dict["status"] = "skipped"
                print("Run skipped")
            else:
                log_dict["status"] = "ok"
                print("Run completed")
        except Exception:
            output_dict = {}
            log_dict["status"] = "failed"
            print("Run failed: " + traceback.format_exc())

        log_dict["total_time"] = int(time.time() - start_time)
        log_dict["output"] = output_dict
        log.append(log_dict)

        # overwrite output log file every time to allow early abort
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
    print("Stopping job")

if __name__ == "__main__":
    main()
