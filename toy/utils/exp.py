import argparse
import traceback
from utils.format import get_leaf_nodes, compare_dicts, namespace2dict, dict2namespace, set_namespace_value


class Exp:
    """
    A class to streamline running multiple experiments with different hyperparameters.
    
    Examples:
    ```python
    from main import main
    from exp import Exp
    from configs.point import hyperparameter_dict_default
    
    exp = Exp(main, hyperparameter_dict_default)
    
    # Run a single experiment
    hyperparameter_dict = hyperparameter_dict_default.copy()
    hyperparameter_dict['data.batch_size'] = 128
    exp.run(hyperparameter_dict)
    
    # Test one hyperparameter
    exp.line_run('data.batch_size', [128, 256, 512], fixed_kv_pairs=[('data.dataset', 'point')])
    
    # Test two hyperparameters
    exp.grid_run('data.batch_size', [128, 256, 512], 'training.lr', [0.001, 0.01, 0.1], fixed_kv_pairs=[('data.dataset', 'point')])
    ```
    """
    def __init__(self, main, hyperparameter_dict_default):
        self.main = main
        self.hyperparameter_dict_default = hyperparameter_dict_default
        self.kvs_default = get_leaf_nodes(hyperparameter_dict_default) # kvs: key-value pairs

    def run(self, input):
        """Run an experiment with given hyperparameters. Wraps main function with automatically-generated experiment name and directory suffix."""
        if isinstance(input, dict):
            hyperparameter_dict = input
        elif isinstance(input, argparse.Namespace):
            hyperparameter_dict = namespace2dict(input)
        else:
            raise ValueError('Input must be a dictionary or an argparse.Namespace')
        kvs = get_leaf_nodes(hyperparameter_dict)
        diff_kvs = compare_dicts(self.kvs_default, kvs) # different key-value pairs
        experiment_name = '_'.join(["{}={}".format(k.split('.')[-1], v) for k, v in diff_kvs.items()])
        
        args = dict2namespace(hyperparameter_dict)
        args.saving.experiment_name = experiment_name
        args.saving.experiment_dir_suffix = experiment_name
        print(f'experiment_name: {experiment_name}')
        try:
            self.main(args)
        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()

    def line_run(self, key, values, fixed_kv_pairs=[]):
        """
        Run multiple experiments changing one hyperparameter.

        Args:
            key (str): Dot-separated string, name of the hyperparameter to be changed, e.g. 'data.batch_size'.
            values (list): Values of the hyperparameter to be tested.
            fixed_kv_pairs (list): Other key-value pairs to be set and then fixed.
        """
        args = dict2namespace(self.hyperparameter_dict_default)
        for k, v in fixed_kv_pairs:
            set_namespace_value(args, k, v)
        for value in values:
            set_namespace_value(args, key, value)
            self.run(args)

    def grid_run(self, key1, values1, key2, values2, fixed_kv_pairs=[]):
        """
        Run multiple experiments changing two hyperparameters.

        Args:
            key1 (str): Dot-separated string, name of the first hyperparameter to be changed, e.g. 'data.batch_size'.
            values1 (list): Values of the first hyperparameter to be tested.
            key2 (str): Dot-separated string, name of the second hyperparameter to be changed, e.g. 'training.lr'.
            values2 (list): Values of the second hyperparameter to be tested.
            fixed_kv_pairs (list of tuples): Other key-value pairs to be set and then fixed, e.g. [('data.dataset', 'cifar10'), ('optimizer.lr', 0.001)].
        """
        args = dict2namespace(self.hyperparameter_dict_default)
        for k, v in fixed_kv_pairs:
            set_namespace_value(args, k, v)
        for value1 in values1:
            for value2 in values2:
                set_namespace_value(args, key1, value1)
                set_namespace_value(args, key2, value2)
                self.run(args)
