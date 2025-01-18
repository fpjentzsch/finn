import json
import numpy as np
import os
import shutil
import torch
import copy
from brevitas.export import export_qonnx
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import (
    GiveRandomTensorNames,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir
from util import summarize_table, summarize_section, power_xml_to_dict, prepare_inputs, delete_dir_contents
from finn.util.test import get_trained_network_and_ishape

from bench_base import bench

def generate_random_threshold_values(
    data_type, num_input_channels, num_steps, narrow=False, per_tensor=False
):
    if per_tensor:
        num_input_channels = 1
    if narrow:
        num_steps -= 1

    return np.random.randint(
        data_type.min(),
        data_type.max() + 1,
        (num_input_channels, num_steps),
    ).astype(np.float32)


def sort_thresholds_increasing(thresholds):
    return np.sort(thresholds, axis=1)

def make_conv_building_block(ifm_dim, ch, kernel_size, simd, pe, parallel_window=0):
    # hardcoded parameters
    idt = DataType["UINT4"]
    wdt = DataType["UINT4"]
    odt = DataType["UINT4"]
    tdt = DataType["UINT32"]
    stride = 1
    in_ch = out_ch = ch  # input channel = output channel for stacking
    # pad so that input dim = output dim for stacking (only supports odd kernel_size for now)
    pad = int(np.floor(kernel_size / 2))

    total_pad = 2 * pad
    out_feature_dim = compute_conv_output_dim(ifm_dim, kernel_size, stride, total_pad)
    weights_shape = [in_ch * kernel_size * kernel_size, out_ch]
    thresholds_shape = [1, odt.get_num_possible_values() - 1]
    input_shape = [1, ifm_dim, ifm_dim, in_ch]
    padding_out_shape = [1, ifm_dim + total_pad, ifm_dim + total_pad, in_ch]
    inpgen_out_shape = [1, out_feature_dim, out_feature_dim, in_ch * kernel_size * kernel_size]
    output_shape = [1, out_feature_dim, out_feature_dim, out_ch]

    assert input_shape == output_shape, "ERROR: Conv layer dimensions not stackable"

    padding_config = {}
    padding_config["domain"] = "finn.custom_op.fpgadataflow.rtl"
    padding_config["backend"] = "fpgadataflow"
    padding_config["ImgDim"] = [ifm_dim, ifm_dim]
    padding_config["NumChannels"] = in_ch
    padding_config["SIMD"] = simd
    padding_config["Padding"] = [pad, pad, pad, pad]
    padding_config["inputDataType"] = idt.name

    inpgen_config = {}
    inpgen_config["domain"] = "finn.custom_op.fpgadataflow.rtl"
    inpgen_config["backend"] = "fpgadataflow"
    inpgen_config["ConvKernelDim"] = [kernel_size, kernel_size]
    inpgen_config["IFMChannels"] = in_ch
    inpgen_config["IFMDim"] = [ifm_dim + total_pad, ifm_dim + total_pad]
    inpgen_config["OFMDim"] = [ifm_dim, ifm_dim]
    inpgen_config["inputDataType"] = idt.name
    inpgen_config["outputDataType"] = idt.name
    inpgen_config["SIMD"] = simd
    inpgen_config["parallel_window"] = parallel_window
    inpgen_config["Stride"] = [stride, stride]
    inpgen_config["Dilation"] = [1, 1]

    mvau_config = {}
    mvau_config["domain"] = "finn.custom_op.fpgadataflow.hls"
    mvau_config["backend"] = "fpgadataflow"
    mvau_config["numInputVectors"] = [1, ifm_dim, ifm_dim]
    mvau_config["MW"] = in_ch * kernel_size * kernel_size
    mvau_config["MH"] = in_ch
    mvau_config["SIMD"] = simd if parallel_window == 0 else simd * kernel_size * kernel_size
    mvau_config["PE"] = pe
    mvau_config["resType"] = "lut"
    mvau_config["mem_mode"] = "internal_embedded"  # internal_decoupled
    mvau_config["inputDataType"] = idt.name
    mvau_config["weightDataType"] = wdt.name
    mvau_config["outputDataType"] = odt.name

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)
    value_info = [
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, weights_shape),
        helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, thresholds_shape),
        helper.make_tensor_value_info("padding_out", TensorProto.FLOAT, padding_out_shape),
        helper.make_tensor_value_info("inpgen_out", TensorProto.FLOAT, inpgen_out_shape),
    ]

    modelproto = qonnx_make_model(
        helper.make_graph(
            name="building_block",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                helper.make_node("FMPadding_rtl", ["top_in"], ["padding_out"], **padding_config),
                helper.make_node(
                    "ConvolutionInputGenerator_rtl",
                    ["padding_out"],
                    ["inpgen_out"],
                    **inpgen_config,
                ),
                helper.make_node(
                    "MVAU_hls", ["inpgen_out", "weights", "thresholds"], ["top_out"], **mvau_config
                ),
            ],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", idt)
    model.set_tensor_layout("top_in", ["N", "H", "W", "C"])
    model.set_tensor_datatype("top_out", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_tensor_datatype("thresholds", tdt)

    weights = gen_finn_dt_tensor(wdt, weights_shape)
    # TODO: thresholds are all the same
    thresholds = generate_random_threshold_values(
        tdt, out_ch, odt.get_num_possible_values() - 1, False, True
    )
    thresholds = sort_thresholds_increasing(thresholds)

    model.set_initializer("weights", weights)
    model.set_initializer("thresholds", thresholds)

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


def combine_blocks(lb, rb, ifm_dim, ch, pe):
    # assumes left branch (lb) and right branch (rb) each have a single (dynamic) input/output with the same shape
    # to avoid mix-ups, start by giving all tensors random names
    lb = lb.transform(GiveRandomTensorNames())
    rb = rb.transform(GiveRandomTensorNames())
    # erase all node names to avoid conflict
    for n in lb.graph.node:
        n.name = ""
    for n in rb.graph.node:
        n.name = ""

    lb_input = lb.graph.input[0]
    lb_output = lb.graph.output[0]
    rb_input = rb.graph.input[0]
    rb_output = rb.graph.output[0]

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ch])
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ch])

    dup_config = {}
    dup_config["domain"] = "finn.custom_op.fpgadataflow.hls"
    dup_config["backend"] = "fpgadataflow"
    dup_config["numInputVectors"] = [1, ifm_dim, ifm_dim]
    dup_config["NumChannels"] = ch
    dup_config["PE"] = pe
    dup_config["NumOutputStreams"] = 2
    dup_config["inputDataType"] = lb.get_tensor_datatype(lb_input.name).name

    add_config = {}
    add_config["domain"] = "finn.custom_op.fpgadataflow.hls"
    add_config["backend"] = "fpgadataflow"
    add_config["numInputVectors"] = [1, ifm_dim, ifm_dim]
    add_config["NumChannels"] = ch
    add_config["PE"] = pe
    add_config["inputDataType"] = lb.get_tensor_datatype(lb_output.name).name

    nodes_lb = [node for node in lb.graph.node]
    nodes_rb = [node for node in rb.graph.node]
    nodes_new = (
        nodes_lb
        + nodes_rb
        + [
            helper.make_node(
                "DuplicateStreams_hls", ["top_in"], [lb_input.name, rb_input.name], **dup_config
            ),
            helper.make_node(
                "AddStreams_hls", [lb_output.name, rb_output.name], ["top_out"], **add_config
            ),
        ]
    )

    value_info_lb = [x for x in lb.graph.value_info]
    value_info_rb = [x for x in rb.graph.value_info]
    value_info_new = value_info_lb + value_info_rb + [lb_input, lb_output, rb_input, rb_output]

    initializer_lb = [x for x in lb.graph.initializer]
    initializer_rb = [x for x in rb.graph.initializer]
    initializer_new = initializer_lb + initializer_rb
    modelproto = qonnx_make_model(
        helper.make_graph(
            name="branching_model",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info_new,
            nodes=nodes_new,
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", lb.get_tensor_datatype(lb_input.name))
    model.set_tensor_layout("top_in", lb.get_tensor_layout(lb_input.name))
    for i in initializer_new:
        model.graph.initializer.append(i)

    # tidy-up
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(GiveReadableTensorNames())
    return model

class bench_fifosizing(bench):
    def step_make_model(self, onnx_export_path):
        np.random.seed(0)
        tmp_output_dir = make_build_dir("tmp_test_fifosizing")

        #TODO: generalize FIFO test so it can be used by other FIFO-related unit tests
        #TODO: allow manual folding/fifo config as input

        #TODO: is a scenario possible where reducing depth of a single FIFO at a time is not sufficient for testing tightness?
        #      e.g. reducing > 1 FIFOs simultaneously does not cause a throughput drop while reducing a single FIFO does?

        #TODO: how to determine rtlsim_n automatically?

        # conv parameters
        dim = self.params["dim"]
        kernel_size = self.params["kernel_size"]
        ch = self.params["ch"]
        simd = self.params["simd"]
        pe = self.params["pe"]
        parallel_window = self.params["parallel_window"]

        lb = None
        for i in range(self.params["lb_num_layers"]):
            new_block = make_conv_building_block(
                dim, ch, kernel_size=kernel_size, simd=simd, pe=pe, parallel_window=parallel_window
            )
            lb = new_block if lb is None else lb.transform(MergeONNXModels(new_block))
        lb.save(tmp_output_dir + "/lb.onnx")

        rb = None
        for i in range(self.params["rb_num_layers"]):
            new_block = make_conv_building_block(
                dim, ch, kernel_size=kernel_size, simd=simd, pe=pe, parallel_window=parallel_window
            )
            rb = new_block if rb is None else rb.transform(MergeONNXModels(new_block))
        rb.save(tmp_output_dir + "/rb.onnx")

        model = combine_blocks(lb, rb, dim, ch, pe=4)
        model.save(onnx_export_path)


    def step_test_fifosizing(self, onnx_export_path, build_dir):
        log = {}
        cfg = build_cfg.DataflowBuildConfig(
            output_dir=build_dir,
            verbose=True, # TODO: remove this?
            # only works with characterization-based FIFO-sizing
            auto_fifo_depths=True,
            auto_fifo_strategy="characterize",
            characteristic_function_strategy=self.params["strategy"],
            split_large_fifos=False,
            # manual folding
            target_fps=None,
            # general rtlsim settings
            force_python_rtlsim=False,
            rtlsim_batch_size=self.params["rtlsim_n"],
            synth_clk_period_ns=self.clock_period_ns,
            board=self.board,
            shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
            generate_outputs=[
                build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                build_cfg.DataflowOutputType.STITCHED_IP,
                build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            ],
        )

        build.build_dataflow_cfg(onnx_export_path, cfg)

        # load performance reports
        with open(build_dir + "/report/estimate_network_performance.json") as f:
            est_data = json.load(f)
        with open(build_dir + "/report/rtlsim_performance.json") as f:
            sim_data = json.load(f) 

        # check for deadlock
        model_final = ModelWrapper(build_dir + "/intermediate_models/step_create_stitched_ip.onnx")
        first_node = getCustomOp(model_final.find_consumer(model_final.graph.input[0].name))
        last_node = getCustomOp(model_final.find_producer(model_final.graph.output[0].name))
        input_txns_expected = np.prod(first_node.get_folded_input_shape()[:-1]) * self.params["rtlsim_n"]
        output_txns_expected = np.prod(last_node.get_folded_output_shape()[:-1]) * self.params["rtlsim_n"]
        deadlock = sim_data["N_IN_TXNS"] != input_txns_expected or sim_data["N_OUT_TXNS"] != output_txns_expected
        log["deadlock"] = deadlock.tolist()

        # check rtlsim throughput
        throughput = sim_data["throughput[images/s]"]
        stable_throughput = sim_data["stable_throughput[images/s]"]
        estimated_throughput = est_data["estimated_throughput_fps"]
        throughput_factor = throughput / estimated_throughput
        stable_throughput_factor = stable_throughput / estimated_throughput

        # TODO: Take throughput or stable_throughput?
        throughput_pass = throughput_factor > self.params["throughput_factor_threshold"]

        log["throughput_pass"] = throughput_pass
        log["throughput"] = throughput
        log["stable_throughput"] = stable_throughput
        log["estimated_throughput"] = estimated_throughput

        # log FIFO sizes for easier inspection
        log["fifo_sizes"] = {}
        for node in model_final.get_nodes_by_op_type("StreamingFIFO_rtl"):
            node_inst = getCustomOp(node)
            log["fifo_sizes"][node.name] = node_inst.get_nodeattr("depth")

        #TODO: log total FIFO sizes (in Bytes)

        # reduce individual FIFO sizes by some amount and observe throughput drop or deadlock appear
        fifo_reduction_pass = []
        log["fifo_reduction_results"] = {}
        model_orig = ModelWrapper(build_dir + "/intermediate_models/step_hw_ipgen.onnx")
        for node_orig in model_orig.get_nodes_by_op_type("StreamingFIFO_rtl"):
            model = copy.deepcopy(model_orig)
            node = model.get_node_from_name(node_orig.name)
            node_inst = getCustomOp(node)

            # skip shallow FIFOs
            # TODO: do we need to consider rounding-up of FIFO depths for impl_style=vivado?
            if node_inst.get_nodeattr("depth") <= self.params["fifo_reduction_skip_threshold"]:
                log["fifo_reduction_results"][node.name] = "skip"
                continue

            # reduce depth of current FIFO and reset generated code
            node_inst.set_nodeattr("depth", int(node_inst.get_nodeattr("depth") * self.params["fifo_reduction_factor"]))
            node_inst.set_nodeattr("code_gen_dir_ipgen", "")
            node_inst.set_nodeattr("ip_path", "")
            node_inst.set_nodeattr("ipgen_path", "")

            # save model variation
            tmp_output_dir_var = build_dir + "/variations/" + node.name
            os.makedirs(tmp_output_dir_var)
            model.save(tmp_output_dir_var + "/model.onnx")

            # build again, only re-run necessary steps to save time
            cfg.output_dir = tmp_output_dir_var
            cfg.steps = ["step_hw_codegen", "step_create_stitched_ip", "step_measure_rtlsim_performance"]
            build.build_dataflow_cfg(tmp_output_dir_var + "/model.onnx", cfg)

            # load performance report
            with open(tmp_output_dir_var + "/report/rtlsim_performance.json") as f:
                sim_data = json.load(f)

            # check for deadlock
            model_final = ModelWrapper(tmp_output_dir_var + "/intermediate_models/step_create_stitched_ip.onnx")
            first_node = getCustomOp(model_final.find_consumer(model_final.graph.input[0].name))
            last_node = getCustomOp(model_final.find_producer(model_final.graph.output[0].name))
            input_txns_expected = np.prod(first_node.get_folded_input_shape()[:-1]) * self.params["rtlsim_n"]
            output_txns_expected = np.prod(last_node.get_folded_output_shape()[:-1]) * self.params["rtlsim_n"]
            var_deadlock = sim_data["N_IN_TXNS"] != input_txns_expected or sim_data["N_OUT_TXNS"] != output_txns_expected

            # check rtlsim throughput
            var_throughput = sim_data["throughput[images/s]"]
            var_stable_throughput = sim_data["stable_throughput[images/s]"]
            # TODO: take throughput or stable_throughput?
            throughput_drop = (throughput - var_throughput) / throughput

            if var_deadlock:   
                fifo_reduction_pass.append(True)
                log["fifo_reduction_results"][node.name] = 1.0
            elif throughput_drop > self.params["fifo_reduction_throughput_drop_threshold"]:
                fifo_reduction_pass.append(True)
                log["fifo_reduction_results"][node.name] = throughput_drop
            else:
                fifo_reduction_pass.append(False)
                log["fifo_reduction_results"][node.name] = "fail (no drop)"
        
        return log

    def run(self):
        # TODO: similar behavior to Transformer DUTs, generalize..
        # Use a temporary dir for buildflow-related files (next to FINN_BUILD_DIR)
        # Ensure it exists but is empty (clear potential artifacts from previous runs)
        tmp_buildflow_dir = os.path.join(os.environ["PATH_WORKDIR"], "buildflow")
        os.makedirs(tmp_buildflow_dir, exist_ok=True)
        delete_dir_contents(tmp_buildflow_dir)
        onnx_export_path = os.path.join(tmp_buildflow_dir, "model_export.onnx")
        build_dir = os.path.join(tmp_buildflow_dir, "build_output")

        self.step_make_model(onnx_export_path)
        self.save_local_artifact("model_step_export", onnx_export_path)

        self.output_dict["fifosizing_testresults"] = self.step_test_fifosizing(onnx_export_path, build_dir)
        self.save_local_artifact("build_output", build_dir) # might be unnecessary to save this

        #TODO: move to base class of bench.py for all DUTs
        if self.debug:
            # Save entire FINN tmp build dir for debugging
            self.save_local_artifact("finn_tmp", os.environ["FINN_BUILD_DIR"])
            self.save_local_artifact("finn_cwd", os.path.join(os.environ["PATH_WORKDIR"], "finn"))
            #TODO: save as early as possible or regardless of errors
