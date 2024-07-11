import itertools
import json
import os
import sys
import time

def consolidate_logs(path, output_filepath):
    log = []
    i = 0
    while (i < 1024):
        if (os.path.isfile(os.path.join(path,"task_%d.json"%(i)))):
            with open(os.path.join(path,"task_%d.json"%(i)), "r") as f:
                log_task = json.load(f)
            log.extend(log_task)
        i = i + 1
    
    with open(output_filepath, "w") as f:
        json.dump(log, f, indent=2)

def merge_logs(log_a, log_b, log_out):
    # merges json log (list of nested dicts) b into a, not vice versa (TODO)

    with open(log_a, "r") as f:
        a = json.load(f)
    with open(log_b, "r") as f:
        b = json.load(f)

    for idx, run_a in enumerate(a):
        for run_b in b:
            if run_a["run_id"] == run_b["run_id"]:
                a[idx] |= run_b
                break

    # also sort by run id
    out = sorted(a, key=lambda x: x["run_id"])

    with open(log_out, "w") as f:
        json.dump(out, f, indent=2)

def wait_for_power_measurements():
    # TODO: make configurable, relative to some env variable due to different mountint points
    bitstreams_path = os.path.join("/mnt/pfs/hpc-prf-radioml/felix/jobs/", 
                            "CI_" + os.environ.get("CI_PIPELINE_IID") + "_" + os.environ.get("CI_PIPELINE_NAME"), 
                            "bench_results/bitstreams")
    
    power_log_path = os.path.join("/mnt/pfs/hpc-prf-radioml/felix/jobs/", 
                            "CI_" + os.environ.get("CI_PIPELINE_IID") + "_" + os.environ.get("CI_PIPELINE_NAME"), 
                            "bench_results/power_measure.json")

    # count bitstreams to measure (can't rely on total number of runs since some of them could've failed)
    files = os.listdir(bitstreams_path)
    bitstream_count = len(list(filter(lambda x : ".bit" in x, files)))

    log = []
    print("Checking if all bitstreams of pipeline have been measured..")
    while(len(log) < bitstream_count):
        if os.path.isfile(power_log_path):
            with open(power_log_path, "r") as f:
                log = json.load(f)
        print("Found measurements for %d/%d bitstreams"%(len(log),bitstream_count))
        time.sleep(60)
    print("Power measurement complete")

if __name__ == "__main__":
    print("Consolidating synthesis results from all sub-jobs of the array")
    consolidate_logs(sys.argv[1], sys.argv[2])

    wait_for_power_measurements()

    print("Merging power measurement logs with remaining logs")
    power_log_path = os.path.join("/mnt/pfs/hpc-prf-radioml/felix/jobs/", 
                            "CI_" + os.environ.get("CI_PIPELINE_IID") + "_" + os.environ.get("CI_PIPELINE_NAME"), 
                            "bench_results/power_measure.json")
    merge_logs(sys.argv[2], power_log_path, sys.argv[2])
    print("Done")
