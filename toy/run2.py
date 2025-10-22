"""Run experiments with customized loops."""
import os
import yaml
from main_light import main
from utils.exp import Exp


with open(os.path.join('configs', 'point.yml'), 'r') as f:
    hyperparameter_dict_default = yaml.load(f, Loader=yaml.SafeLoader)

# Customize default hyperparameters
hyperparameter_dict_default['saving']['result_dir']='results112'
hyperparameter_dict_default['training']['model_load_path']=os.path.join("model_weights", "scorenet_20_0.01_8.pth")
hyperparameter_dict_default['sampling']['log_freq']=100
exp = Exp(main, hyperparameter_dict_default)

# Run experiment loops
"""
Example usage:

# Iterate over one hyperparameter
exp.line_run('sampling.k_d', [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])

# Iterate over two hyperparameters
exp.grid_run('sampling.k_d', [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'sampling.k_i', [0.0, 0.1, 0.2, 0.3, 0.5],)

# Iterate over two hyperparameters and fix one hyperparameter
exp.grid_run('sampling.k_d', [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'sampling.k_i', [0.0, 0.1, 0.2, 0.3, 0.5], fixed_kv_pairs=[('sampling.k_i_decay', 1.0)])
"""
print('Done.')
