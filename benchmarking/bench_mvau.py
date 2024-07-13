import copy
import json
import math
import numpy as np
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
    roundup_to_integer_multiple,
)
from shutil import copy as shcopy

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.make_zynq_proj import collect_ip_dirs
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.util.basic import make_build_dir, pynq_native_port_width, pynq_part_map

# power report scripting based on Lucas Reuter:
template_open = """
open_project  $PROJ_PATH$
open_run $RUN$
"""

template_single_test = """
set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ -hier -type lut [get_cells -r finn_design_i/.*]
set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ -hier -type register [get_cells -r finn_design_i/.*]
set_switching_activity -deassert_resets
report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
reset_switching_activity -hier -type lut [get_cells -r finn_design_i/.*]
reset_switching_activity -hier -type register [get_cells -r finn_design_i/.*]
"""

# template_single_test_type = """
# set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ -hier -type $SWITCH_TARGET$ [get_cells -r finn_design_i/.*]
# set_switching_activity -deassert_resets
# report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
# reset_switching_activity -hier -type $SWITCH_TARGET$ [get_cells -r finn_design_i/.*]
# """

template_sim_power = """
set_property SOURCE_SET sources_1 [get_filesets sim_1]
import_files -fileset sim_1 -norecurse $TB_FILE_PATH$
set_property top switching_simulation_tb [get_filesets sim_1]
update_compile_order -fileset sim_1

launch_simulation -mode post-implementation -type functional
restart
open_saif $SAIF_FILE_PATH$
log_saif [get_objects -r /switching_simulation_tb/dut/*]
run $SIM_DURATION_NS$ ns
close_saif

read_saif $SAIF_FILE_PATH$
report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
"""

# TODO: configurable clock frequency
template_switching_simulation_tb = """
`timescale 1 ns/10 ps

module switching_simulation_tb;
reg clk;
reg rst;

//dut inputs
reg tready;
reg [$INSTREAM_WIDTH$-1:0] tdata;
reg tvalid;

//dut outputs
wire [$OUTSTREAM_WIDTH$-1:0] accel_tdata;
wire accel_tready;
wire accel_tvalid;

finn_design_wrapper dut(
        .ap_clk(clk),
        .ap_rst_n(rst),
        .m_axis_0_tdata(accel_tdata),
        .m_axis_0_tready(tready),
        .m_axis_0_tvalid(accel_tvalid),
        .s_axis_0_tdata(tdata),
        .s_axis_0_tready(accel_tready),
        .s_axis_0_tvalid(tvalid)
        );

always
    begin
        clk = 0;
        #2.5;
        clk = 1;
        #2.5;
    end

integer i;
initial
    begin
        tready = 0;
        tdata = 0;
        tvalid = 0;
        rst = 0;
        #50;
        rst = 1;
        tvalid = 1;
        tready = 1;
        while(1)
            begin
                for (i = 0; i < $INSTREAM_WIDTH$/$DTYPE_WIDTH$; i = i+1) begin
                    tdata[i*$DTYPE_WIDTH$ +: $DTYPE_WIDTH$] = $RANDOM_FUNCTION$;
                end
                #5;
            end
    end
endmodule
"""

zynq_harness_template = """
set FREQ_MHZ %s
set NUM_AXILITE %d
if {$NUM_AXILITE > 9} {
    error "Maximum 10 AXI-Lite interfaces supported"
}
set NUM_AXIMM %d
set BOARD %s
set FPGA_PART %s
create_project finn_zynq_link ./ -part $FPGA_PART

# set board part repo paths to find boards installed by FINN
set paths_prop [get_property BOARD_PART_REPO_PATHS [current_project]]
set paths_param [get_param board.repoPaths]
lappend paths_prop $::env(FINN_ROOT)/deps/board_files
lappend paths_param $::env(FINN_ROOT)/deps/board_files
set_property BOARD_PART_REPO_PATHS $paths_prop [current_project]
set_param board.repoPaths $paths_param

if {$BOARD == "RFSoC2x2"} {
    set_property board_part xilinx.com:rfsoc2x2:part0:1.1 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} else {
    puts "Unrecognized board"
}

create_bd_design "top"
if {$ZYNQ_TYPE == "zynq_us+"} {
    set zynq_ps_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:zynq_ultra_ps_e:*"]]
    create_bd_cell -type ip -vlnv $zynq_ps_vlnv zynq_ps
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ps]
    set_property CONFIG.PSU__DISPLAYPORT__PERIPHERAL__ENABLE {0} [get_bd_cells zynq_ps]
    #activate one slave port, deactivate the second master port
    set_property -dict [list CONFIG.PSU__USE__S_AXI_GP2 {0}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0}] [get_bd_cells zynq_ps]
    #set frequency of PS clock (this can't always be exactly met)
    set_property -dict [list CONFIG.PSU__OVERRIDE__BASIC_CLOCK {0}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ [expr int($FREQ_MHZ)]] [get_bd_cells zynq_ps]
} else {
    puts "Unrecognized Zynq type"
}

#instantiate axi interconnect, axi smartconnect
set interconnect_vlnv [get_property VLNV [get_ipdefs -all "xilinx.com:ip:axi_interconnect:*" -filter design_tool_contexts=~*IPI*]]
#set smartconnect_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:smartconnect:*"]]
create_bd_cell -type ip -vlnv $interconnect_vlnv axi_interconnect_0
#create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_0
#set number of axilite interfaces, and number of axi master interfaces
#set_property -dict [list CONFIG.NUM_SI $NUM_AXIMM] [get_bd_cells smartconnect_0]
set_property -dict [list CONFIG.NUM_MI $NUM_AXILITE] [get_bd_cells axi_interconnect_0]

#create reset controller and connect interconnects to PS
if {$ZYNQ_TYPE == "zynq_us+"} {
    set axi_peripheral_base 0xA0000000
    #connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]
    connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/S00_AXI]
    #connect interconnect clocks and resets
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/S00_ACLK]
    #apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
}
#connect_bd_net [get_bd_pins axi_interconnect_0/ARESETN] [get_bd_pins smartconnect_0/aresetn]

#procedure used by below IP instantiations to map BD address segments based on the axi interface aperture
proc assign_axi_addr_proc {axi_intf_path} {
    #global variable holds current base address
    global axi_peripheral_base
    #infer range
    set range [expr 2**[get_property CONFIG.ADDR_WIDTH [get_bd_intf_pins $axi_intf_path]]]
    set range [expr $range < 4096 ? 4096 : $range]
    #align base address to range
    set offset [expr ($axi_peripheral_base + ($range-1)) & ~($range-1)]
    #perform assignment
    assign_bd_address [get_bd_addr_segs $axi_intf_path/Reg*] -offset $offset -range $range
    #advance base address
    set axi_peripheral_base [expr $offset + $range]
}

#custom IP instantiations/connections start here
%s

#finalize clock and reset connections for interconnects
if {$ZYNQ_TYPE == "zynq_us+"} {
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} }  [get_bd_pins axi_interconnect_0/M*_ACLK]
}

save_bd_design
assign_bd_address
validate_bd_design

set_property SYNTH_CHECKPOINT_MODE "Hierarchical" [ get_files top.bd ]
make_wrapper -files [get_files top.bd] -import -fileset sources_1 -top

#set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
#set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
#set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
#set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]
#set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
#set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
#set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
#set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]

# out-of-context synth can't be used for bitstream generation
# set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} -objects [get_runs synth_1]
launch_runs -to_step write_bitstream impl_1
wait_on_run [get_runs impl_1]

# generate synthesis report
open_run impl_1
report_utilization -hierarchical -hierarchical_depth 4 -file synth_report.xml -format xml
close_project
"""


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


def _find_rows_and_headers(table):
    rows = table.findall("tablerow")
    headers = []

    for row in rows:
        headers = row.findall("tableheader")
        if len(headers) > 0:
            break
    return (rows, headers)


def summarize_table(table):
    table_summary = {}
    table_summary["headers"] = []
    rows, headers = _find_rows_and_headers(table)

    if len(headers) > 0:
        string = "Header: "
        for header in headers:
            table_summary["headers"].append(header.attrib["contents"])
            string = string + header.attrib["contents"] + " "
        # print(string.rstrip())

    for row in rows:
        cells = row.findall("tablecell")
        if len(cells) > 0:
            cell_name = cells[0].attrib["contents"]
            string = cell_name
            table_summary[cell_name] = []
            for cell in cells[1:]:
                table_summary[cell_name].append(cell.attrib["contents"])
                string = string + cell.attrib["contents"] + " "
            # print(string.rstrip())

    return table_summary


def summarize_section(section):
    section_summary = {}
    section_summary["tables"] = []
    section_summary["subsections"] = {}

    # print("Section:", section.attrib["title"])
    tables = section.findall("table")
    sub_sections = section.findall("section")
    for table in tables:
        section_summary["tables"].append(summarize_table(table))
    # print("")
    for sub_section in sub_sections:
        section_summary["subsections"][sub_section.attrib["title"]] = summarize_section(sub_section)

    return section_summary


def power_xml_to_dict(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sections = root.findall("section")
    result = {}

    for section in sections:
        result[section.attrib["title"]] = summarize_section(section)

    return result


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


def make_single_mvau_model(
    W,
    numInputVectors,
    pe,
    simd,
    m,
    wdt,
    idt,
    odt,
    T=None,
    tdt=None,
    mem_mode="const",
    ram_style="auto",
    ram_style_thresholds="auto",
):
    mw = W.shape[0]
    mh = W.shape[1]

    # there are two ways to implement bipolar weights and inputs for
    # MatrixVectorActivation:
    # - specify their datatypes as such
    # - specify their datatypes as BINARY as use binaryXnorMode
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # we'll internally convert weights/inputs to binary and specify the
        # datatypes as such, and also set the binaryXnorMode attribute to 1
        export_wdt = DataType["BINARY"]
        export_idt = DataType["BINARY"]
        binary_xnor_mode = 1
    else:
        export_wdt = wdt
        export_idt = idt
        binary_xnor_mode = 0

    # numInputVectors for dense = [N]
    # numInputVectors for conv  = [N, H, W]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, numInputVectors + [mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, numInputVectors + [mh])
    if T is not None:
        no_act = 0
        node_inp_list = ["inp", "weights", "thresh"]
        if odt == DataType["BIPOLAR"]:
            actval = 0
        else:
            actval = odt.min()
    else:
        # no thresholds
        node_inp_list = ["inp", "weights"]
        actval = 0
        no_act = 1
    mvau_node = helper.make_node(
        "MVAU_hls",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        M=m,
        numInputVectors=numInputVectors,
        inputDataType=export_idt.name,
        weightDataType=export_wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
        resType="lut",
        mem_mode=mem_mode,
        ram_style=ram_style,
        ram_style_thresholds=ram_style_thresholds,
        runtime_writeable_weights=0,
    )

    graph = helper.make_graph(nodes=[mvau_node], name="mvau_graph", inputs=[inp], outputs=[outp])
    model = qonnx_make_model(graph, producer_name="mvau-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    # model.set_tensor_shape("weights", (channels, 1, k_h, k_w)) from VVAU
    if binary_xnor_mode:
        # convert bipolar to binary
        model.set_initializer("weights", (W + 1) / 2)
    else:
        model.set_initializer("weights", W)
    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)

    # Minimize weight & accumulator width to obtain realistic resource consumption
    # model = model.transform(InferShapes())
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(InferDataTypes())

    return model


def prepare_inputs(input_tensor, idt, wdt):
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert bipolar to binary
        return {"inp": (input_tensor + 1) / 2}
    else:
        return {"inp": input_tensor}


def bench_mvau(params, task_id, run_id, results_dir, save_dir):
    part = "xczu28dr-ffvg1517-2-e"  # TODO: make configurable, + Alveo support?
    clock_period_ns = 5

    # Read params
    idt = params["idt"]
    wdt = params["wdt"]
    act = params["act"]

    numInputVectors = params["nhw"]
    mw = params["mw"]
    mh = params["mh"]
    sf = params["sf"]
    nf = params["nf"]
    m = params["m"]

    mem_mode = params["mem_mode"]
    ram_style = params["ram_style"]
    ram_style_thr = params["ram_style_thr"]

    do_hls = params["do_hls"]
    do_rtlsim = params["do_rtlsim"]
    do_synthesis = params["do_synthesis"]
    do_sim_power = params["do_sim_power"]
    do_synth_power = params["do_synth_power"]

    if "dut_duplication" in params:
        dut_duplication = params["dut_duplication"]
    else:
        dut_duplication = 1

    output_dict = {}

    # convert string to FINN DataType
    idt = DataType[idt]
    wdt = DataType[wdt]
    if act is not None:
        act = DataType[act]

    # Determine and log folding
    if sf == -1:
        sf = mw
    simd = mw // sf
    if nf == -1:
        nf = mh
    pe = mh // nf
    if mw % simd != 0 or mh % pe != 0:
        print("Invalid simd/pe configuration, skipping")
        return
    if m > 1 and (simd != mw or pe != mh):
        print("M > 1 not possible for non-max simd/pe, skipping")
        return
    output_dict["simd"] = simd
    output_dict["pe"] = pe

    # Generate weights
    np.random.seed(123456)  # TODO: verify or switch to modern numpy random generation

    W = gen_finn_dt_tensor(wdt, (mw, mh))

    if "sparsity_type" in params:
        sparsity_type = params["sparsity_type"]
    else:
        sparsity_type = "none"

    if sparsity_type == "none":
        if "sparsity_amount" in params:
            if params["sparsity_amount"] > 0:
                print("sparsity amount > 0 not applicable for none sparsity, skipping")
                return
    else:
        if params["sparsity_amount"] == 0:
            print("sparsity amount = 0 not applicable for selected sparsity, skipping")
            return
        if sparsity_type == "unstructured":
            idx = np.random.choice(
                mw * mh, size=int(params["sparsity_amount"] * mw * mh), replace=False
            )
            W = np.reshape(W, -1)
            W[idx] = 0.0
            W = np.reshape(W, (mw, mh))
        elif sparsity_type == "rows_random":
            idx_mw = np.random.choice(mw, size=int(params["sparsity_amount"] * mw), replace=False)
            W[idx_mw, :] = 0.0
        elif sparsity_type == "cols_random":
            idx_mh = np.random.choice(mh, size=int(params["sparsity_amount"] * mh), replace=False)
            W[:, idx_mh] = 0.0
        elif sparsity_type == "rows_regular":
            if params["sparsity_amount"] == 0.25:
                idx_mw = np.arange(0, mw, step=4)
            elif params["sparsity_amount"] == 0.5:
                idx_mw = np.arange(0, mw, step=2)
            elif params["sparsity_amount"] == 0.75:
                idx_mw = np.concatenate(
                    (np.arange(0, mw, step=4), np.arange(1, mw, step=4), np.arange(2, mw, step=4))
                )
            else:
                print("regular sparsity only applicable for amount 0.25/0.5/0.75, skipping")
                return
            W[idx_mw, :] = 0.0
        elif sparsity_type == "cols_regular":
            if params["sparsity_amount"] == 0.25:
                idx_mh = np.arange(0, mh, step=4)
            elif params["sparsity_amount"] == 0.5:
                idx_mh = np.arange(0, mh, step=2)
            elif params["sparsity_amount"] == 0.75:
                idx_mh = np.concatenate(
                    (np.arange(0, mh, step=4), np.arange(1, mh, step=4), np.arange(2, mh, step=4))
                )
            else:
                print("regular sparsity only applicable for amount 0.25/0.5/0.75, skipping")
                return
            W[:, idx_mh] = 0.0

        else:
            print("ERROR: unknown sparsity type")
            raise Exception("ERROR: unknown sparsity type")

    # TODO: implement enforce option which prevents naturally occurring sparsity
    # params["sparsity_enforce"]
    # TODO: implement distribution option which selects between uniform/normal/??
    # params["sparsity_distribution"]

    # log resulting sparsity statistics
    # could be higher than selected due to naturally occurring sparsity
    num_zeros = (W == 0).sum()
    num_ones = (W == 1).sum() + (W == -1).sum()
    num_p2 = 0
    for w in np.nditer(W):
        if w != 0 and w != 1 and w != -1:
            if w > 0:
                if math.log2(w).is_integer():
                    num_p2 = num_p2 + 1
            else:
                if math.log2(-w).is_integer():
                    num_p2 = num_p2 + 1
    output_dict["zero_weights"] = round(num_zeros / W.size, 2)
    output_dict["easy_weights"] = round((num_zeros + num_ones + num_p2) / W.size, 2)

    # Generate thresholds
    if act is None:
        # no activation, produce accumulators
        T = None
        tdt = None
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            odt = DataType["UINT32"]
        else:
            odt = DataType["INT32"]
    else:
        odt = act
        # set range for threshold values according to worst-case accumulator range (not weight value specific)
        # this could result in some thresholds being clipped by MinimizeAccumulatorWidth
        # lower_range = calculate_matvec_accumulator_range(wdt.min() * np.ones_like(W), idt)
        # upper_range = calculate_matvec_accumulator_range(wdt.max() * np.ones_like(W), idt)
        # acc_min = min(min(lower_range), min(upper_range))
        # acc_max = max(max(lower_range), max(upper_range))
        # set range for threshold values according to actual accumulator range for the generated weights
        (acc_min, acc_max) = calculate_matvec_accumulator_range(W, idt)
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(acc_min, acc_max - 1, (mh, n_steps)).astype(np.float32)
        # provide non-decreasing thresholds
        T = np.sort(T, axis=1)
        # generate thresholds for activation
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            tdt = DataType["UINT32"]
            # bias thresholds to be positive
            T = np.ceil((T + mw) / 2)
            assert (T >= 0).all()
        else:
            tdt = DataType["INT32"]

    # Create model
    model = make_single_mvau_model(
        W, numInputVectors, pe, simd, m, wdt, idt, odt, T, tdt, mem_mode, ram_style, ram_style_thr
    )
    model = model.transform(GiveUniqueNodeNames())
    node = model.get_nodes_by_op_type("MVAU_hls")[0]
    inst = getCustomOp(node)

    # Save model
    results_dir_models = os.path.join(results_dir, "models")
    os.makedirs(results_dir_models, exist_ok=True)
    model.save(os.path.join(results_dir_models, "model_%d_initial.onnx" % (run_id)))

    # Gather FINN estimates
    print("Gathering FINN estimates")
    finn_resources_model = res_estimation(model, fpgapart=part)
    finn_resources = finn_resources_model[node.name]
    finn_cycles_model = model.analysis(exp_cycles_per_layer)
    finn_cycles = finn_cycles_model[node.name]

    finn_estimates = finn_resources
    finn_estimates["CYCLES"] = finn_cycles
    output_dict["finn_estimates"] = finn_estimates

    # Perform Vitis HLS synthesis for HLS resource/performance reports
    if do_hls:
        start_time = time.time()
        print("Performing Vitis HLS synthesis")
        model = model.transform(PrepareIP(part, clock_period_ns))
        model = model.transform(HLSSynthIP())

        hls_resources_model = model.analysis(hls_synth_res_estimation)
        hls_resources = hls_resources_model[node.name]
        output_dict["hls_estimates"] = hls_resources
        output_dict["hls_time"] = int(time.time() - start_time)

        model_hlssynth = copy.deepcopy(model)

    # Perform RTL simulation for performance measurement
    if do_rtlsim:
        start_time = time.time()
        print("Performing Verilator RTL simulation (n=1)")
        # Prepare
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
        # Generate input data
        x = gen_finn_dt_tensor(idt, numInputVectors + [mw])
        input_dict = prepare_inputs(x, idt, wdt)
        # Run
        oxe.execute_onnx(model, input_dict)["outp"]  # do not check output for correctness
        # Log result
        node = model.get_nodes_by_op_type("MVAU_hls")[0]
        inst = getCustomOp(node)
        rtlsim_cycles = inst.get_nodeattr("cycles_rtlsim")
        output_dict["rtlsim_cycles"] = rtlsim_cycles
        output_dict["rtlsim_time"] = int(time.time() - start_time)

    # Perform Vivado synthesis for accurate resource/timing and inaccurate power reports
    if do_synthesis:
        start_time = time.time()
        print("Performing Vivado (stitched-ip, out-of-context) synthesis")
        model = model.transform(ReplaceVerilogRelPaths())
        model = model.transform(CreateStitchedIP(part, clock_period_ns))
        model = model.transform(SynthOutOfContext(part=part, clk_period_ns=clock_period_ns))
        ooc_synth_results = eval(model.get_metadata_prop("res_total_ooc_synth"))

        # Naive power reports
        results_dir_power = os.path.join(results_dir, "power", "run_%d" % (run_id))
        os.makedirs(results_dir_power, exist_ok=True)
        start_test_batch_fast(
            results_path=results_dir_power,
            project_path=os.path.join(
                ooc_synth_results["vivado_proj_folder"], "vivadocompile", "vivadocompile.xpr"
            ),
            run_target="impl_1",
            pairs=[(25, 0.5), (50, 0.5), (75, 0.5)],
        )

        # Log most important power results directly (refer to detailed logs for more)
        for reportname in ["25_0.5", "50_0.5", "75_0.5"]:
            with open(os.path.join(results_dir_power, "%s.json" % reportname), "r") as f:
                report = json.load(f)
                power = float(report["Summary"]["tables"][0]["Total On-Chip Power (W)"][0])
                power_dyn = float(report["Summary"]["tables"][0]["Dynamic (W)"][0])
                ooc_synth_results["power_%s" % reportname] = power
                ooc_synth_results["power_dyn_%s" % reportname] = power_dyn

        output_dict["ooc_synth"] = ooc_synth_results
        output_dict["ooc_synth_time"] = int(time.time() - start_time)

    # Perform Vivado simulation for accurate power report
    if do_sim_power:
        start_time = time.time()
        print("Performing Vivado simulation for power report")
        if do_rtlsim:
            sim_duration_ns = output_dict["rtlsim_cycles"] * 3 * clock_period_ns
        else:
            sim_duration_ns = output_dict["finn_estimates"]["CYCLES"] * 3 * clock_period_ns
        sim_power_report(
            results_path=results_dir_power,
            project_path=os.path.join(
                ooc_synth_results["vivado_proj_folder"], "vivadocompile", "vivadocompile.xpr"
            ),
            in_width=inst.get_instream_width(),
            out_width=inst.get_outstream_width(),
            dtype_width=idt.bitwidth(),
            sim_duration_ns=sim_duration_ns,
        )

        # Log most important power results directly (refer to detailed logs for more)
        for reportname in ["sim"]:
            with open(os.path.join(results_dir_power, "%s.json" % reportname), "r") as f:
                report = json.load(f)
                power = float(report["Summary"]["tables"][0]["Total On-Chip Power (W)"][0])
                power_dyn = float(report["Summary"]["tables"][0]["Dynamic (W)"][0])
                output_dict["power_%s" % reportname] = power
                output_dict["power_dyn%s" % reportname] = power_dyn

        output_dict["sim_power_time"] = int(time.time() - start_time)

    # Perform Vivado synthesis for on-hardware power measurement
    if do_synth_power:
        start_time = time.time()
        print("Performing Vivado synthesis with test harness integration for power measurement")

        model = model_hlssynth.transform(ReplaceVerilogRelPaths())
        model = model.transform(CreateStitchedIP(part, clock_period_ns))
        model = model.transform(
            MakeZYNQHarnessProject(
                platform="RFSoC2x2",
                results_dir=results_dir,
                save_dir = save_dir,
                run_id=run_id,
                dut_duplication=dut_duplication,
                clock_period_ns=clock_period_ns
            )
        )

        output_dict["synth_power_time"] = int(time.time() - start_time)

    model.save(os.path.join(results_dir_models, "model_%d_final.onnx" % (run_id)))

    return output_dict
