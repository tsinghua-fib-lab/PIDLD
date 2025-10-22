import argparse
import json
from types import NoneType

import numpy as np


def dict2namespace(config):
    """Recursively convert a nested dictionary to a argparse.Namespace object. Reference: https://github.com/ermongroup/ncsnv2/blob/master/main.py, line 155."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(ns):
    """Recursively convert a Namespace object to a regular dictionary."""
    if isinstance(ns, argparse.Namespace):
        d = vars(ns)
        return {k: namespace2dict(v) for k, v in d.items()}
    elif isinstance(ns, dict):
        return {k: namespace2dict(v) for k, v in ns.items()}
    elif isinstance(ns, list):
        return [namespace2dict(x) for x in ns]
    else:
        return ns


def get_namespace_value(namespace, keys):
    """Get the value of a nested namespace with a series of keys."""
    value = namespace
    for key in keys:
        value = getattr(value, key)
    return value


def set_namespace_value(namespace, key:str, value):
    """Set the value of a nested namespace **in-place** with a series of keys.
    
    Args:
        namespace (argparse.Namespace): The namespace to be modified.
        keys (str): A string of keys separated by '.', e.g. "model.num_layers"
        value: The value to be set.
    """
    ks = key.split('.') # Keys, e.g. ['model', 'num_layers']
    for k in ks[:-1]:
        namespace = getattr(namespace, k)
    setattr(namespace, ks[-1], value)
    return namespace


def get_leaf_nodes(d:dict, parent_key=''):
    """Recursively get all leaf nodes in a nested dictionary."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(get_leaf_nodes(v, new_key))
        else:
            items.update({new_key: v})
    return items


def compare_dicts(dict1, dict2):
    """Compare the key-value pairs of two dictionaries and return the different ones."""
    assert dict1.keys() == dict2.keys(), "Keys must be the same"

    different_kv_pairs = {}
    for key in dict1.keys():
        value1 = dict1[key]
        value2 = dict2[key]
        assert type(value1) in [int, float, str, bool, list, tuple, NoneType], "Unsupported value type: {}".format(type(value1))
        
        if value1!=value2:
            different_kv_pairs.update({key: value2})

    return different_kv_pairs


class NumpyEncoder(json.JSONEncoder):
    """Customized json encoder for numpy array data."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Convert to list for json serialization.
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)
