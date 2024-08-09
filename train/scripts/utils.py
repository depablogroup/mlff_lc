import yaml

def read_yaml_file(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)


def write_yaml_file(file_path, data):
    with open(file_path, 'w') as f:
        yaml.dump(data, f)


def compare_dicts(dict1, dict2, tolerance=1e-6):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    if keys1 != keys2:
        return False

    for key in keys1:
        value1 = dict1[key]
        value2 = dict2[key]

        if isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                return False

            for v1, v2 in zip(value1, value2):
                if isinstance(v1, float) and isinstance(v2, float):
                    if abs(v1 - v2) > tolerance:
                        return False
                elif v1 != v2:
                    return False
        elif isinstance(value1, float) and isinstance(value2, float):
            if abs(value1 - value2) > tolerance:
                return False
        elif value1 != value2:
            return False

    return True
