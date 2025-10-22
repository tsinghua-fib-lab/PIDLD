"""Run experiments with command line arguments."""
import os
import copy
import argparse
import yaml
from main import main
from utils.exp import Exp


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--result_dir', type=str, default='results', help='directory to save results')
parser.add_argument('--model_load_path', type=str, default=r"model_weights\scorenet_20_0.01_8.pth", help='path to model weights')
parser.add_argument('-P', '--k_p', type=float, default=None, help='value of k_p')
parser.add_argument('-I', '--k_i', type=float, default=None, help='value of k_i')
parser.add_argument('-D', '--k_d', type=float, default=None, help='value of k_d')
parser.add_argument('--k_i_decay', type=float, default=None, help='value of k_i_decay')
args = parser.parse_args()

with open(os.path.join('configs', 'point.yml'), 'r') as f:
    hyperparameter_dict_default = yaml.load(f, Loader=yaml.FullLoader)
hyperparameter_dict_default['saving']['result_dir'] = args.result_dir
hyperparameter_dict_default['training']['model_load_path'] = args.model_load_path
exp = Exp(main, hyperparameter_dict_default)

hyperparameter_dict=copy.copy(hyperparameter_dict_default)
# If any of the command line arguments is provided, update the hyperparameter_dict accordingly; otherwise, use the default value in the config file.
if args.seed != None:
    hyperparameter_dict['seed'] = args.seed
if args.k_p != None:
    hyperparameter_dict['sampling']['k_p'] = args.k_p
if args.k_i != None:
    hyperparameter_dict['sampling']['k_i'] = args.k_i
    print(f'k_i: {hyperparameter_dict["sampling"]["k_i"]}')
if args.k_d != None:
    hyperparameter_dict['sampling']['k_d'] = args.k_d
if args.k_i_decay != None:
    hyperparameter_dict['sampling']['k_i_decay'] = args.k_i_decay
    
exp.run(hyperparameter_dict)

print('Done.')
