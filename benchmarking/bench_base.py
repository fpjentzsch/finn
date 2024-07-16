import itertools
import os
import subprocess
import copy
import json
import time
import traceback
import glob
from shutil import copy as shcopy
from shutil import copytree
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
from templates import template_open, template_single_test, template_sim_power, template_switching_simulation_tb, zynq_harness_template
from util import summarize_table, summarize_section, power_xml_to_dict, prepare_inputs
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from qonnx.util.basic import (
    gen_finn_dt_tensor,
    roundup_to_integer_multiple,
)
import pandas as pd
import onnxruntime as ort

class MakeZYNQHarnessProject(Transformation):
    """Based on MakeZYNQProject transformation, but integrates IP into test harness instead of DMA shell."""

    def __init__(self, platform, artifacts_dir, separate_bistream_dir, run_id, dut_duplication=1, clock_period_ns=10):
        super().__init__()
        self.platform = platform
        self.artifacts_dir = artifacts_dir
        self.separate_bistream_dir = separate_bistream_dir
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
        artifacts_dir_bitstreams = os.path.join(self.artifacts_dir, "bitstreams")
        os.makedirs(artifacts_dir_bitstreams, exist_ok=True)

        bitfile_name = vivado_pynq_proj_dir + "/finn_zynq_link.runs/impl_1/top_wrapper.bit"
        if not os.path.isfile(bitfile_name):
            raise Exception(
                "Synthesis failed, no bitfile found. Check logs under %s" % vivado_pynq_proj_dir
            )
        deploy_bitfile_name = artifacts_dir_bitstreams + "/run_%d.bit" % self.run_id
        shcopy(bitfile_name, deploy_bitfile_name)

        hwh_name = vivado_pynq_proj_dir + "/finn_zynq_link.gen/sources_1/bd/top/hw_handoff/top.hwh"
        if not os.path.isfile(hwh_name):
            raise Exception(
                "Synthesis failed, no hwh file found. Check logs under %s" % vivado_pynq_proj_dir
            )
        deploy_hwh_name = artifacts_dir_bitstreams + "/run_%d.hwh" % self.run_id
        shcopy(hwh_name, deploy_hwh_name)

        synth_report_filename = vivado_pynq_proj_dir + "/synth_report.xml"
        deploy_synth_report_filename = artifacts_dir_bitstreams + "/run_%d.xml" % self.run_id
        shcopy(synth_report_filename, deploy_synth_report_filename)

        # copy additionally to a separate PFS location for measurement automation to pick it up
        # TODO: make this more configurable or switch to job/artifact based power measurement 
        shcopy(deploy_bitfile_name, self.separate_bistream_dir)
        shcopy(deploy_hwh_name, self.separate_bistream_dir)
        shcopy(deploy_synth_report_filename, self.separate_bistream_dir)

        clock_period_mhz = int(1.0 / self.clock_period_ns * 1000.0)
        measurement_settings = {"freq_mhz": clock_period_mhz}
        with open(os.path.join(self.separate_bistream_dir, "run_%d_settings.json"%self.run_id), "w") as f:
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
    def __init__(self, params, task_id, run_id, artifacts_dir, save_dir):
        super().__init__()
        self.params = params
        self.task_id = task_id
        self.run_id = run_id
        self.artifacts_dir = artifacts_dir
        self.save_dir = save_dir

        # General configuration
        self.board = "RFSoC2x2"
        self.part = "xczu28dr-ffvg1517-2-e"  # TODO: make configurable, + Alveo support?
        self.clock_period_ns = 10

        # Initialize output directories (might exist from other runs of the same job)
        self.artifacts_dir_models = os.path.join(self.artifacts_dir, "models")
        os.makedirs(self.artifacts_dir_models, exist_ok=True)
        self.artifacts_dir_power = os.path.join(self.artifacts_dir, "power_vivado", "run_%d" % (self.run_id))
        os.makedirs(self.artifacts_dir_power, exist_ok=True)

        self.save_dir_bitstreams = os.path.join(self.save_dir, "bitstreams")
        os.makedirs(self.save_dir_bitstreams, exist_ok=True)

        # Intermediate models saved between steps
        # TODO: create setter functions for intermediate models or other artifacts that log them to gitlab artifacts or local dir automatically
        self.model_initial = None
        self.model_step_hls = None
        self.model_step_synthesis = None

        # Initialize dictionary to collect all benchmark results
        self.output_dict = {}

    def save_artifact(self, name, source_path):
        target_path = os.path.join(self.artifacts_dir, name, "run_%d" % (self.run_id))
        os.makedirs(target_path, exist_ok=True)
        if os.path.isdir(source_path):
            copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shcopy(source_path, target_path)

    def save_local_artifact(self, name, source_path):
        target_path = os.path.join(self.save_dir, name, "run_%d" % (self.run_id))
        os.makedirs(target_path, exist_ok=True)
        if os.path.isdir(source_path):
            copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shcopy(source_path, target_path)

    def step_make_model(self):
        # may be implemented in subclass
        pass

    def step_export_onnx(self):
        # may be implemented in subclass
        pass

    def step_build(self):
        # may be implemented in subclass
        pass

    def run(self):
        # must be implemented in subclass
        pass

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
        input_shape = model.get_tensor_shape(input_tensor.name)
        input_dtype = model.get_tensor_datatype(input_tensor.name)
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
            results_path=self.artifacts_dir_power,
            project_path=os.path.join(
                ooc_synth_results["vivado_proj_folder"], "vivadocompile", "vivadocompile.xpr"
            ),
            run_target="impl_1",
            pairs=[(25, 0.5), (50, 0.5), (75, 0.5)],
        )

        # Log most important power results directly (refer to detailed logs for more)
        for reportname in ["25_0.5", "50_0.5", "75_0.5"]:
            with open(os.path.join(self.artifacts_dir_power, "%s.json" % reportname), "r") as f:
                report = json.load(f)
                power = float(report["Summary"]["tables"][0]["Total On-Chip Power (W)"][0])
                power_dyn = float(report["Summary"]["tables"][0]["Dynamic (W)"][0])
                ooc_synth_results["power_%s" % reportname] = power
                ooc_synth_results["power_dyn_%s" % reportname] = power_dyn

        self.output_dict["ooc_synth"] = ooc_synth_results
        self.output_dict["ooc_synth_time"] = int(time.time() - start_time)

        # Save model for logging purposes
        model.save(os.path.join(self.artifacts_dir_models, "model_%d_synthesis.onnx" % (self.run_id)))
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
        input_node_inst = getCustomOp(model.find_consumer(input_tensor.name))
        output_node_inst = getCustomOp(model.find_producer(output_tensor.name))
        sim_power_report(
            results_path=self.artifacts_dir_power,
            project_path=os.path.join(
                self.output_dict["ooc_synth"]["vivado_proj_folder"], "vivadocompile", "vivadocompile.xpr"
            ),
            in_width=input_node_inst.get_instream_width(),
            out_width=output_node_inst.get_outstream_width(),
            dtype_width=model.get_tensor_datatype(input_tensor.name).bitwidth(),
            sim_duration_ns=sim_duration_ns,
        )

        # Log most important power results directly (refer to detailed logs for more)
        for reportname in ["sim"]:
            with open(os.path.join(self.artifacts_dir_power, "%s.json" % reportname), "r") as f:
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
                artifacts_dir=self.artifacts_dir,
                separate_bistream_dir=self.save_dir_bitstreams,
                run_id=self.run_id,
                dut_duplication=dut_duplication,
                clock_period_ns=self.clock_period_ns
            )
        )

        self.output_dict["synth_power_time"] = int(time.time() - start_time)

        # Save model for logging purposes
        model.save(os.path.join(self.artifacts_dir_models, "model_%d_synth_power.onnx" % (self.run_id)))

    def step_parse_builder_output(self, build_dir):
        # Used to parse selected reports/logs into the output json dict for DUTs that use a full FINN builder flow

        # CHECK FOR VERIFICATION STEP SUCCESS
        # Collect all verification output filenames
        outputs = glob.glob(os.path.join(build_dir, "verification_output/*.npy"))
        # Extract the verification status for each verification output by matching
        # to the SUCCESS string contained in the filename
        status = all([
            out.split("_")[-1].split(".")[0] == "SUCCESS" for out in outputs
        ])
   
        # Construct a dictionary reporting the verification status as string
        self.output_dict["builder_verification"] = {"verification": {True: "success", False: "fail"}[status]}
        # TODO: mark job as failed if verification fails

        # PARSE LOGS
        report_path = os.path.join(build_dir, "report/post_synth_resources.json")
        report_filter = "(top)"
        # Open the report file
        with open(report_path) as file:
            # Load the JSON formatted report
            report = pd.read_json(file, orient="index")
        # Filter the reported rows according to some regex filter rule
        report = report.filter(regex=report_filter, axis="rows")
        # Generate a summary of the total resources
        summary = report.sum()

        #TODO: parse finn estimates, hls estimates, step times, (rtlsim n=1, n=100)
        #TODO: add vivado latency simulation for special transformer case
        
        self.output_dict["builder"] = summary.to_dict()

    def steps_simple_model_flow(self):
        # Default step sequence for benchmarking a simple model (mostly single operators/custom_ops)
        do_hls = self.params["do_hls"] if "do_hls" in self.params else False
        do_rtlsim = self.params["do_rtlsim"] if "do_rtlsim" in self.params else False
        do_synthesis = self.params["do_synthesis"] if "do_synthesis" in self.params else False
        do_sim_power = self.params["do_sim_power"] if "do_sim_power" in self.params else False
        do_synth_power = self.params["do_synth_power"] if "do_synth_power" in self.params else False

        # Perform steps
        model, dut_info = self.step_make_model()

        # Save model for logging purposes
        # TODO: benchmarking infrastructure could be integrated deeper into ONNX IR and FINN custom_op/transformation infrastructure
        # E.g. parameters and paths could be stored as onnx attributes and benchmarking steps as generic or specialized custom_op transformations
        model.save(os.path.join(self.artifacts_dir_models, "model_%d_initial.onnx" % (self.run_id)))

        # Save model for use in other steps
        self.model_initial = model

        # Log dict reported by DUT-specific scripts to overall result dict
        # E.g. this could contain SIMD/PE derived from folding factors or weight distribution information
        self.output_dict["info"] = dut_info

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

    def steps_full_build_flow(self):
        # Default step sequence for benchmarking a full FINN builder flow
        onnx_export_path = "model_export.onnx"
        build_dir = "build_output"
        #TODO: put in isolated temp dirs?

        self.step_export_onnx(onnx_export_path)
        self.save_local_artifact("model_step_export", onnx_export_path)

        self.step_build(onnx_export_path, build_dir)
        self.save_local_artifact("build_output", build_dir)

        self.step_parse_builder_output(build_dir)
