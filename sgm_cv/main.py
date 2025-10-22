"""Customized main function for PID ALD sampling on image generation task"""
import os
import sys
import time
import argparse
import logging
import traceback
import json

import numpy as np
import torch
import seaborn as sns
import yaml

from dynamics import PID_ALD
from models.refinenet import RefineNet
from models.ema import EMAHelper
from evaluation.inception import fid_inception_v3
from utils import set_seed
from utils.format import dict2namespace, NumpyEncoder
from utils.log import get_logger, close_logger
from utils.hooks import SavingHook, VisualizationHook, RecordingHook, EvaluationHook


sns.set_theme(style="whitegrid")


def parse_args_and_config():
    
    # Load hyperparameters
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--exp_name', type=str, default='default', help="Experiment name")
    parser.add_argument('--exp_dir_suffix', type=str, default=None, help="Suffix for the experiment directory")

    parser.add_argument('-P', '--k_p', type=float, default=None, help="Coefficient for Proportional Gain")
    parser.add_argument('-I', '--k_i', type=float, default=None, help="Coefficient for Integral Gain")
    parser.add_argument('-D', '--k_d', type=float, default=None, help="Coefficient for Differential Gain")
    parser.add_argument('--k_i_decay', type=float, default=None, help="Decay rate for Integral Gain")
    parser.add_argument('--k_d_decay', type=float, default=None, help="Decay rate for Differential Gain")
    parser.add_argument('-T', '--n_steps_each', type=int, default=None, help="Number of sampling steps per noise level")
    parser.add_argument('-L', '--num_classes', type=int, default=None, help="Number of noise levels")

    args = parser.parse_args() # args: arguments that are more likely to be changed

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config) # config: arguments that are less likely to be changed

    # Update experiment arguments using default configurations
    if args.k_p is not None:
        config.sampling.k_p = args.k_p
    if args.k_i is not None:
        config.sampling.k_i = args.k_i
    if args.k_d is not None:
        config.sampling.k_d = args.k_d
    if args.k_i_decay is not None:
        config.sampling.k_i_decay = args.k_i_decay
    if args.k_d_decay is not None:
        config.sampling.k_d_decay = args.k_d_decay
    if args.n_steps_each is not None:
        config.sampling.n_steps_each = args.n_steps_each
    if args.num_classes is not None:
        config.model.num_classes = args.num_classes
    
    return args, config


def main(args, config):

    # Set random seeds
    set_seed(args.seed)

    # Set up experiment directory
    time_string = str(int(time.time())) # Time string to identify the current experiment
    experiment_dir = os.path.join(args.exp, 'experiment_{}_{}_{}_{}'.format(time_string, config.sampling.k_p, config.sampling.k_i, config.sampling.k_d))
    if args.exp_dir_suffix is None:
        experiment_dir += '_' + args.exp_name
    elif args.exp_dir_suffix:
        experiment_dir += '_' + args.exp_dir_suffix
    os.makedirs(experiment_dir, exist_ok=False)
    print('Experiment directory created at {}.'.format(experiment_dir))
    image_dir = os.path.join(experiment_dir, 'image_samples')
    os.makedirs(image_dir)
    print('Image directory created at {}.'.format(image_dir))

    # Set up the root logger
    logger = get_logger(os.path.join(experiment_dir, 'stdout.txt'))

    try:
        
        # Save experiment args and config
        with open(os.path.join(experiment_dir, 'config.yml'), 'w') as f:
            yaml.dump(vars(config), f, default_flow_style=False, sort_keys=False)
        with open(os.path.join(experiment_dir, 'args.yml'), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)

        # Record experiment information
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info("Exp directory: '{}'".format(experiment_dir))
        logging.info("Exp instance id = {}".format(os.getpid()))
        logging.info("Exp comment = {}".format(args.comment))
        logging.info("Using device: {}".format(device))

        # Load score model
        sigmas = torch.tensor(
                    np.exp(
                        np.linspace(
                            np.log(config.model.sigma_begin),
                            np.log(config.model.sigma_end),
                            config.model.num_classes
                            )
                        )
                    ).float().to(device)
        sigmas_np = sigmas.cpu().numpy()

        score = RefineNet(ngf=config.model.ngf, num_classes=config.model.num_classes, sigmas=sigmas, data_channels=config.data.channels).to(device)
        score = torch.nn.DataParallel(score, device_ids=[0])

        # Initialize the score model with the pre-trained weights
        states = torch.load(config.model.score_model_state_dict_path, map_location=device, weights_only=True) # Load pretrained score model
        logging.info("Loaded score model state dictionary from '{}'.".format(config.model.score_model_state_dict_path))
        states[0]['module.sigmas'] = sigmas
        states[-1]['module.sigmas'] = sigmas
        logging.info("Updated the sigmas in the state dictionary.")
        logging.info("sigma_begin: {}; sigma_end: {}; number of sigmas: {}.".format(sigmas[0], sigmas[-1], len(sigmas)))

        score.load_state_dict(states[0], strict=True)
        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)
            logging.info("Used EMA helper to update score model parameters.")
        logging.info('Number of score model parameters: {}.'.format(sum([p.numel() for p in score.parameters() if p.requires_grad]))) # 29694083

        del states

        # Load inception model and statistics for FID calculation
        inception_v3_model = fid_inception_v3().to(device)
        inception_v3_model.eval()
        logging.info('Number of inception model parameters: {}.'.format(sum([p.numel() for p in inception_v3_model.parameters() if p.requires_grad]))) # 23850960
        
        inception_stats=np.load(config.evaluation.inception_stats_path) # Downloaded from http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz
        logging.info("Inception stats loaded from '{}'.".format(config.evaluation.inception_stats_path))
        mu_real = inception_stats['mu']
        sigma_real = inception_stats['sigma']

        # Generate samples
        all_init_samples = torch.rand(config.sampling.n_samples, config.data.channels,
                                        config.data.image_size, config.data.image_size,
                                        device=device) # Typically (10000,3,32,32)
        score.eval() # Important!!
        
        recording_hook = RecordingHook(verbose=config.sampling.verbose)
        saving_hook = SavingHook(save=config.sampling.save, freq=config.sampling.freq, last_only=config.sampling.last_only,
                                    sample_save_dir=image_dir, verbose=config.sampling.verbose)
        visualization_hook = VisualizationHook(save=config.visualization.save, freq=config.visualization.freq, last_only=config.visualization.last_only,
                                                nrow=config.visualization.nrow, sample_save_dir=image_dir, verbose=config.visualization.verbose)
        evaluation_hook = EvaluationHook(inception_v3_model=inception_v3_model, mu_real=mu_real, sigma_real=sigma_real, device=device,
                                        batch_size=config.evaluation.batch_size, num_workers=config.evaluation.num_workers,
                                        evaluate=config.evaluation.evaluate, freq=config.evaluation.freq, last_only=config.evaluation.last_only,
                                        verbose=config.evaluation.verbose)

        final_samples = PID_ALD(all_init_samples,
                                    scorenet=score, sigmas=sigmas_np, n_steps_each=config.sampling.n_steps_each, step_lr=config.sampling.step_lr,
                                    k_p=config.sampling.k_p, k_i=config.sampling.k_i, k_d=config.sampling.k_d,
                                    k_i_decay=config.sampling.k_i_decay, k_d_decay=config.sampling.k_d_decay,
                                    device=device, batch_size=config.sampling.batch_size,
                                    saving_hook=saving_hook, visualization_hook=visualization_hook,
                                    recording_hook=recording_hook, evaluation_hook=evaluation_hook,
                                    denoise=config.sampling.denoise, verbose=config.sampling.verbose,
        )

        sampler_record_dict_save_path = os.path.join(experiment_dir,'sampler_record_dict.json')
        json.dump(recording_hook.sampler_record_dict, open(sampler_record_dict_save_path, 'w'), indent=4, cls=NumpyEncoder)
        logging.info("Sampler record dict saved to '{}'.".format(sampler_record_dict_save_path))

        metric_record_dict_save_path = os.path.join(experiment_dir,'metric_record_dict.json')
        json.dump(evaluation_hook.metric_record_dict, open(metric_record_dict_save_path, 'w'), indent=4, cls=NumpyEncoder)
        logging.info("Metric record dict saved to '{}'.".format(metric_record_dict_save_path))

        close_logger(logger)
        return 0
    
    except:
        logging.error(traceback.format_exc())
        close_logger(logger)
        return 1


if __name__ == '__main__':
    sys.exit(main(*parse_args_and_config()))
