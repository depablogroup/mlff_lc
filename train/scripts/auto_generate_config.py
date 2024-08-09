"""Generate a series of configs based on a lightweight yml config file.

Args:
    template: the template allegro config to use for the training. By default,
        it will search the file in ../templates/, but you can override this
        behavior by providing the full path.
    config: the config file that includes the fields that will be replaced from
        the template. 
Sample Usage:
python auto_generate_config.py --template lc_smaller.yaml \
--config change_cutoff_small.yaml --seriesdir change_cutoff_small \
--crossproduct
"""
import argparse
import yaml
import os
import itertools

import utils

def generate_combinations(replacement_dict):
    keys = list(replacement_dict.keys())
    values = list(replacement_dict.values())
    combinations = list(itertools.product(*values))
    return keys, combinations


def generate_sequential_combinations(replacement_dict):
    keys = list(replacement_dict.keys())
    values = list(replacement_dict.values())
    
    # Check if all lists have the same length or are of length 1
    lengths = {len(value) for value in values if len(value) > 1}
    if len(lengths) > 1:
        raise ValueError("Inconsistent list lengths in replacement values.")
    
    length = lengths.pop() if lengths else 1
    
    combinations = []
    for i in range(length):
        combination = []
        for value in values:
            combination.append(value[i] if len(value) > 1 else value[0])
        combinations.append(tuple(combination))

    return keys, combinations


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--crossproduct", action="store_true")
    parser.add_argument("--seriesdir", type=str, required=True)
    return parser.parse_args()


def main(args: argparse.Namespace):
    # By default, find the files in the corresponding directory
    template_path = os.path.join("../templates/", args.template)
    replacement_path = os.path.join("../configs/", args.config)
    
    template_data = utils.read_yaml_file(template_path)
    replacement_data = utils.read_yaml_file(replacement_path)

    # Check the keys
    if "run_name" in replacement_data:
        raise ValueError("`run_name` will be automatically populated.")

    if "root" in replacement_data:
        if len(replacement_data["root"]) > 1:
            raise ValueError("Experiments in one batch should share the same root. Abort.")
    else:
        raise ValueError("No root specified for the batch of experiments. Abort.")
    os.makedirs(replacement_data["root"][0], exist_ok=True)

    if args.crossproduct:
        keys, combinations = generate_combinations(replacement_data)
    else:
        keys, combinations = generate_sequential_combinations(replacement_data)

    for key in keys:
        if key not in template_data:
            raise ValueError(f"`{key}` is not provided in the template. Abort.")

    output_dir = os.path.join("../config_series/", args.seriesdir)  # Output directory to save new files
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving the new configs into {output_dir}.")

    for combination in combinations:
        new_data = template_data.copy()
        filename_suffix = ''
        
        for key, value in zip(keys, combination):
            new_data[key] = value
            if key == "root":
                continue
            value_str = value
            if key == "dataset_file_name":
                continue  # Shorten the filenames

            if isinstance(value, list) or isinstance(value, tuple):
                value_str = "_".join(str(item) for item in value)
            filename_suffix += f"{key}_{value_str}_"
        
        filename_suffix = filename_suffix.rstrip("_")
        run_name = filename_suffix
        new_data["run_name"] = run_name
        new_file_path = os.path.join(output_dir, f"{filename_suffix}.yaml")
        utils.write_yaml_file(new_file_path, new_data)

        # Check the validity of the new data
        generated_data = utils.read_yaml_file(new_file_path)
        if not utils.compare_dicts(new_data, generated_data):
            raise ValueError(f"Generated data at {new_file_path} is not the same as intended.")
        print(f"Generated file: {new_file_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)
