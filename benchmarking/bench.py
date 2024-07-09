import itertools
import json
import os
import sys
import time
import traceback
from bench_mvau import bench_mvau
from bench_rtl_swg import bench_rtl_swg


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
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "results", "task_%d.json" % (task_id))
    print("Collecting results in path: %s" % results_dir)

    # TODO: support multiple, populate from available configs (+ optional CI variables?)
    benchmark_select = "MVAU_hls"  # sys.argv[1]
    config_select = "MVAU_hls.json"  # sys.argv[2]

    # Select benchmark
    print("Running benchmark %s" % (benchmark_select))
    if benchmark_select == "ConvolutionInputGenerator_rtl":
        bench = bench_rtl_swg
    elif benchmark_select == "MVAU_hls":
        bench = bench_mvau
    else:
        print("ERROR: benchmark not found")
        return

    # Select config (given relative to this script)
    config_path = os.path.join(os.path.dirname(__file__), "cfg", config_select)
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

    # Save config (only first job of array)
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
            output_dict = bench(params, task_id, run_id, results_dir)
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
