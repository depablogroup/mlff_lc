"""Automatically submit allegro training jobs to slurm. 

Currently, does not support multiple runs, but APIs are provided.
"""
import argparse
import textwrap
import model_info
import os
import yaml
import subprocess
import re
import numpy as np
import pandas as pd


SUBMISSION_TEMPLATE = re.compile(r"Submitted batch job ([0-9]+)\n")

CLUSTER_INFO_DICT = {
    "apple": model_info.ClusterInfo(
        slurm_template=textwrap.dedent(
            """\
            #!/bin/bash
            #SBATCH -t 1:00:00
            #SBATCH --nodes=1
            #SBATCH --partition=apple
            #SBATCH --gres=gpu:1
            #SBATCH --job-name={job_identifier}
            {python_cmds}
            """
        ),
        ntasks_per_node=1
    ),
    "pear": model_info.ClusterInfo(
        slurm_template=textwrap.dedent(
            """\
            #!/bin/bash
            #SBATCH -t 1:00:00
            #SBATCH --nodes=1
            #SBATCH --partition=pear
            #SBATCH --gres=gpu:1
            #SBATCH --job-name={job_identifier}
            {python_cmds}
            """
        ),
        ntasks_per_node=1,
    ),
    "banana": model_info.ClusterInfo(
        slurm_template=textwrap.dedent(
            """\
            #!/bin/bash
            #SBATCH -t 1:00:00
            #SBATCH --nodes=1
            #SBATCH --partition=banana
            #SBATCH --gres=gpu:1
            #SBATCH --job-name={job_identifier}
            {python_cmds}
            """
        ),
        ntasks_per_node=1,
    ),
}

JOB_TEMPLATE = ("NEQUIP_NUM_TASKS=8 \n"
                "nequip-train {config_path} --warn-unused")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seriesdir", type=str, required=True)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--cluster", type=str, required=True)
    parser.add_argument("--username", type=str, default="anonymous")
    parser.add_argument("--newexponly", action="store_true")
    return parser.parse_args()


def get_run_info(run_dir: str, run_info: model_info.RunInfo):
    config_path = os.path.join(run_dir, "config.yaml")
    metrics_path = os.path.join(run_dir, "metrics_epoch.csv")

    if not os.path.exists(config_path) or not os.path.exists(metrics_path):
        run_info.run_status = run_info.RunStatus.FAILED
        return run_info

    with open(config_path, "r") as config_file:
        config_data = yaml.load(config_file, Loader=yaml.Loader)

    try:
        metrics_data = pd.read_csv(metrics_path, sep=",\s*", engine="python")
    except pd.errors.EmptyDataError:
        run_info.run_status = run_info.RunStatus.NEED_RERUN
        print(f"{run_dir} needs rerun")
        return run_info
    run_info.model_record = metrics_data

    final_epoch = metrics_data["epoch"].iloc[-1]
    final_wall = metrics_data["wall"].iloc[-1]
    final_lr = metrics_data["LR"].iloc[-1]
    validation_loss_col = metrics_data["validation_loss"]
    run_info.best_val_loss = np.min(validation_loss_col)

    patience = config_data["early_stopping_patiences"]["validation_loss"]
    if patience > 0:
        last_losses = validation_loss_col.iloc[-patience:]
        if all(loss >= last_losses.iloc[0] for loss in last_losses):
            run_info.run_status = run_info.RunStatus.FINISHED
            return run_info

    if (final_epoch < config_data["max_epochs"] and
        final_wall < config_data["early_stopping_upper_bounds"]["cumulative_wall"] and
        final_lr > config_data["early_stopping_lower_bounds"]["LR"]):
        run_info.run_status = run_info.RunStatus.NEED_RERUN
        print(f"{run_dir} needs rerun")
    else:
        run_info.run_status = run_info.RunStatus.FINISHED
        print(f"{run_dir} has already finished")
    return run_info


def get_static_job_status(config_series_dir: str):
    """Get the current, static Job Status"""
    job_status = {}
    run_names = []
    roots = set()
    for filename in os.listdir(config_series_dir):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            file_path = os.path.join(config_series_dir, filename)
            with open(file_path, "r") as file:
                data = yaml.load(file, Loader=yaml.Loader)
                run_name = data.get("run_name")
                root = data.get("root")

                if run_name is not None:
                    run_names.append(run_name)
                if root is not None:
                    roots.add(root)

    if len(roots) != 1:
        raise ValueError("The directory contains configs of multiple roots. Abort.")
    root = roots.pop()

    for run_name in run_names:
        # For each config, create a run status
        run_info = model_info.RunInfo()
        run_dir = os.path.join(root, run_name)
        if not os.path.exists(run_dir):
            run_info.run_status = model_info.RunInfo.RunStatus.NOT_STARTED
        else:
            get_run_info(run_dir, run_info)
        job_status[run_name] = run_info

    return job_status, run_names, root


def get_ongoing_job_status(job_status, run_names, username: str="anonymous"):
    """Get the status of how many shards are currently ongoing."""
    try:
        cmd = ["squeue", "-u", username, "-h", "-a", "-t", "RUNNING,PENDING,COMPLETING", "-o", "%A %j"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
        job_info_lines = result.stdout.strip().split('\n')
        for line in job_info_lines:
            if not line:
                continue  # Skip empty line
            job_id, job_name = line.split()
            if job_name.startswith("tr_al_"):
                run_name = job_name.split(":")[-1]
                if run_name in run_names:
                    job_status[run_name].run_status = model_info.RunInfo.RunStatus.ONGOING

    except subprocess.CalledProcessError as e:
        print("Error when checking the current jobs")
        raise e
    return job_status


def decide_which_job_to_run(job_status, run_names, newexponly=False):
    run_jobs = []
    for run_name in run_names:
        run_info = job_status[run_name]
        if newexponly and run_info.run_status == model_info.RunInfo.RunStatus.NOT_STARTED:
            run_jobs.append(run_name)
            print(f"{run_name} scheduled.")
        elif newexponly:
            continue
        elif (run_info.run_status == model_info.RunInfo.RunStatus.ONGOING
            or run_info.run_status == model_info.RunInfo.RunStatus.FINISHED):
            continue  # No need to run
        elif run_info.run_status == model_info.RunInfo.RunStatus.FAILED:
            print(f"The config {run_name} failed.")
        else:
            run_jobs.append(run_name)
            print(f"{run_name} scheduled.")
    return run_jobs


def prepare_single_slurm_script(
        cluster: str,
        out_dir: str,
        job_name: str
    ):
    """Prepare a single slurm script."""
    cluster_info = CLUSTER_INFO_DICT[cluster]
    slurm_template = cluster_info.slurm_template
    config_path = os.path.join(out_dir, f"{job_name}.yaml")
    config_path = os.path.abspath(config_path)
    python_str = ""
    python_str += JOB_TEMPLATE.format(
        config_path=config_path
    )
    slurm_script = slurm_template.format(job_identifier=job_name,
                                         python_cmds=python_str,
                                         out_dir=out_dir)
    return slurm_script


def prepare_scripts(
        cluster: str,
        jobs_to_run: list,
        out_dir: str,
    ):
    slurm_scripts = []
    for _, job_name in enumerate(jobs_to_run):
        slurm_script = prepare_single_slurm_script(cluster, out_dir, job_name)
        slurm_scripts.append(slurm_script)
    return slurm_scripts


def submit_script(
        slurm_script: str,
        out_dir: str,
        script_index: int,
        dry_run: bool=False
):
    # Create a temporary file in the out_dir to generate the script
    temp_script_path = os.path.join(out_dir, f"temp_{script_index}.sh")
    temp_script_path = os.path.abspath(temp_script_path)
    with open(temp_script_path, "w") as fp:
        fp.write(slurm_script)
    if not dry_run:
        try:
            result = subprocess.run(["sbatch", temp_script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Job submission failed: {e}")
            raise e
        output = result.stdout.decode('UTF-8')
        print(output)
        jobid = re.search(SUBMISSION_TEMPLATE, output)
    else:
        print(f"{temp_script_path} created")
        jobid = None
    return jobid


def submit_scripts(
        slurm_scripts,
        out_dir: str,
        dry_run: bool=False,
    ):
    jobids = []
    for i, slurm_script in enumerate(slurm_scripts):
        jobid = submit_script(slurm_script=slurm_script,
                              out_dir=out_dir,
                              script_index=i,
                              dry_run=dry_run)
        jobids.append(jobid)
    return jobids


def main(args: argparse.Namespace):
    config_series_dir = os.path.join("../config_series/", args.seriesdir)
    cluster = args.cluster
    dry_run = args.dryrun
    username = args.username
    newexponly = args.newexponly

    job_status, run_names, _ = get_static_job_status(config_series_dir)
    job_status = get_ongoing_job_status(job_status, run_names, username)
    print("Job status fetched")
    jobs_to_run = decide_which_job_to_run(job_status, run_names, newexponly)
    print(f"Shard to evaluate decided: {len(jobs_to_run)} in total")
    slurm_scripts = prepare_scripts(
        cluster=cluster, 
        jobs_to_run=jobs_to_run,
        out_dir=config_series_dir
    )
    print(f"Slurm scripts generated: {len(slurm_scripts)} in total")
    submit_scripts(slurm_scripts, config_series_dir, dry_run=dry_run)
    if not dry_run:
        print("Slurm job submitted")
    else:
        print("This is a dry run, no job submitted.")


if __name__ == "__main__":
    args = get_args()
    main(args)
