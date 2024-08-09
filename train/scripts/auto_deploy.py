"""Automatically deploy trained allegro models into force fields.
"""

import argparse
import textwrap
import model_info
import os
import subprocess
import re

import numpy as np

import auto_submit

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


JOB_TEMPLATE = ("nequip-deploy build --train-dir {train_dir} "
                "{model_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seriesdir", type=str, required=True)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--cluster", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--username", type=str, default="anonymous")
    parser.add_argument("--converged_only", action="store_true")
    return parser.parse_args()


def decide_which_job_to_run(job_status, run_names, converged_only=False):
    run_jobs = []
    best_run_name = None
    best_val_loss = np.inf
    for run_name in run_names:
        run_info = job_status[run_name]
        if run_info.run_status == model_info.RunInfo.RunStatus.FINISHED:
            run_jobs.append(run_name)
            print(f"{run_name} scheduled and converged.")
            if run_info.best_val_loss < best_val_loss:
                best_run_name = run_name
                best_val_loss = run_info.best_val_loss
        elif run_info.run_status == model_info.RunInfo.RunStatus.NEED_RERUN and not converged_only:
            run_jobs.append(run_name)
            print(f"{run_name} scheduled, but this model can be improved by further training.")
            if run_info.best_val_loss < best_val_loss:
                best_run_name = run_name
                best_val_loss = run_info.best_val_loss
        elif run_info.run_status == model_info.RunInfo.RunStatus.FAILED:
            print(f"The config {run_name} failed.")
    # Check if the best validation loss is way too bad
    print(f"The best run is {run_name}")
    if best_val_loss > 1e-3:
        print(f"WARNING: The best validation loss is {best_val_loss:.4f} > 1e-3.")
    return run_jobs, best_run_name


def prepare_single_python_str(
        out_dir: str,
        job_name: str
    ):
    """Prepare a single slurm script."""
    python_str = ""
    train_dir = os.path.join(out_dir, job_name)
    model_path = os.path.join(train_dir, "model.pth")
    python_str += JOB_TEMPLATE.format(
        train_dir=train_dir,
        model_path=model_path
    )
    return python_str


def copy_best_checkpoint(best_run_name:str, out_dir:str):
    move_str = "cp {source} {dest}"
    source = os.path.join(out_dir, best_run_name, "model.pth")
    dest = os.path.join(out_dir, "model.pth")
    return move_str.format(source=source, dest=dest)


def prepare_scripts(
        cluster: str,
        jobs_to_run: list,
        out_dir: str,
        best_run_name: str,
    ):
    cluster_info = CLUSTER_INFO_DICT[cluster]
    slurm_template = cluster_info.slurm_template
    python_strs = []
    for _, job_name in enumerate(jobs_to_run):
        python_str = prepare_single_python_str(out_dir, job_name)
        python_strs.append(python_str)
    concat_python_str = "\n".join(python_strs)
    job_identifier = f"auto_deploy_{os.path.basename(out_dir)}"
    slurm_script = slurm_template.format(job_identifier=job_identifier, python_cmds=concat_python_str)
    slurm_scripts = [slurm_script]
    return slurm_scripts


def main(args):
    config_series_dir = os.path.join("../config_series/", args.seriesdir)
    cluster = args.cluster
    dry_run = args.dryrun
    username = args.username
    converged_only = args.converged_only
    job_status, run_names, result_dir = auto_submit.get_static_job_status(config_series_dir)
    job_status = auto_submit.get_ongoing_job_status(job_status, run_names, username)
    result_dir = os.path.abspath(result_dir)
    print(f"Job status fetched, results stored in {result_dir}")
    jobs_to_run, best_run_name = decide_which_job_to_run(job_status, run_names, converged_only=converged_only)
    print(f"Shard to evaluate decided: {len(jobs_to_run)} in total")
    slurm_scripts = prepare_scripts(
        cluster=cluster, 
        jobs_to_run=jobs_to_run,
        out_dir=result_dir,
        best_run_name=best_run_name,
    )
    print("Slurm scripts generated")
    jobids = auto_submit.submit_scripts(slurm_scripts, result_dir, dry_run=dry_run)
    if not dry_run:
        print("Slurm job submitted")
    else:
        print("This is a dry run, no job submitted.")
    return


if __name__ == "__main__":
    args = get_args()
    main(args)
