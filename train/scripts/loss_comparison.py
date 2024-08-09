"""Compare the loss curve of a series of experiments, and visualize it in figure.
"""
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

def common_prefix(strings):
    if not strings:
        return ""
    
    prefix = strings[0]
    for string in strings[1:]:
        i = 0
        while i < len(prefix) and i < len(string) and prefix[i] == string[i]:
            i += 1
        prefix = prefix[:i]
    return prefix


def common_suffix(strings):
    if not strings:
        return ""
    
    suffix = strings[0]
    for string in strings[1:]:
        i = -1
        while abs(i) <= len(suffix) and abs(i) <= len(string) and suffix[i] == string[i]:
            i -= 1
        suffix = suffix[i+1:] if i != -1 else suffix
    return suffix


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="validation_loss")
    args = parser.parse_args()
    args.resultsdir = os.path.join("../results/", args.resultsdir)
    return args


def fetch_training_data(datadir: str):
    """Obtain the training data for an experiment."""
    training_data_path = os.path.join(datadir, "metrics_epoch.csv")
    training_data = pd.read_csv(training_data_path, sep=",\s*", engine="python")
    return training_data


def fetch_training_data_series(seriesdir: str):
    subdirs = os.listdir(seriesdir)
    subdirs.sort()
    training_data_series = {}
    for subdir in subdirs:
        if not os.path.isdir(os.path.join(seriesdir, subdir)):
            continue
        # Filter out the dataset
        if subdir.startswith("processed_dataset"):
            continue
        training_data = fetch_training_data(os.path.join(seriesdir, subdir))
        training_data_series[subdir] = training_data
    return training_data_series


def generate_plot(training_data_series: dict, metric: str, seriesdir: str):
    """Generate a summary plot in the seriesdir."""
    keys = list(training_data_series.keys())

    prefix = common_prefix(keys)
    suffix = common_suffix(keys)
    prefix_len = len(prefix)
    # suffix_len = len(suffix)
    suffix_len = 0

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
    for config, training_data in training_data_series.items():
        epoch = training_data["epoch"]
        recorded_metric = training_data[metric]
        unique_key = config[prefix_len:len(config) - suffix_len]
        ax.plot(epoch, recorded_metric, label=unique_key)
    ax.legend()
    ax.set_title(os.path.basename(seriesdir))
    fig.savefig(os.path.join(seriesdir, f"{metric}.png"))


def main(args: argparse.Namespace):
    training_data_series = fetch_training_data_series(args.resultsdir)
    generate_plot(training_data_series, args.metric, args.resultsdir)


if __name__ == "__main__":
    args = get_args()
    main(args)
