"""light-weight version of main.py"""
import os
import time
import json
import logging
import traceback
from collections import deque

import numpy as np
import torch
import tqdm
import yaml

from utils import set_seed, get_act
from utils.format import dict2namespace, namespace2dict, NumpyEncoder
from utils.log import get_logger, close_logger
from utils.metrics import gmm_estimation, gmm_kl
from models.simple_models import SimpleNet1d

os.environ["OMP_NUM_THREADS"] = "5" # To avoid the warning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.


def main(args):
    
    try:
        
        # Directory Setup
        if not os.path.exists(args.saving.result_dir):
            os.makedirs(args.saving.result_dir, exist_ok=True) # The directory to store all experiment results
            print("Result directory created at {}.".format(args.saving.result_dir))
        
        time_string = str(int(time.time())) # Time string to identify the experiment
        experiment_dir = os.path.join(
                            args.saving.result_dir,
                            'experiment_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                time_string,
                                str(args.model.sigma_begin),
                                str(args.model.sigma_end),
                                str(args.model.num_classes),
                                str(args.sampling.n_steps_each),
                                str(args.sampling.k_p),
                                str(args.sampling.k_i),
                                str(args.sampling.k_d),
                                str(args.sampling.k_i_decay),
                                str(args.sampling.k_d_decay),
                                ))
        if args.saving.experiment_dir_suffix != '':
            experiment_dir += '_' + args.saving.experiment_dir_suffix
        elif args.saving.experiment_dir_suffix == '':
            experiment_dir += '_' + args.saving.experiment_name

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir) # The directory to store the results of the current experiment
            print("Experiment directory created at {}.".format(experiment_dir))

        # Logging Setup
        log_file_path = os.path.join(experiment_dir, 'log.log') # Set the log path
        logger = get_logger(log_file_path=log_file_path) # Set and get the root logger
        logging.info("Experiment directory: '{}'".format(experiment_dir))

        # Configuration Saving
        
        config_dict = namespace2dict(args)
        config_save_path = os.path.join(experiment_dir, 'config.json')
        json.dump(config_dict, open(config_save_path, 'w'), indent=4, cls=NumpyEncoder)
        logging.info("Experiment config saved to '{}'.".format(config_save_path))

    except Exception as e:
        print("Error: {}".format(str(e)))
        return 1


    try: # Now the logger has been successfully set up, and errors can be logged in the log file.

        set_seed(args.seed)

        # Noise Scale Generation
        sigmas = torch.tensor(
                    np.exp(
                        np.linspace(
                            np.log(args.model.sigma_begin),
                            np.log(args.model.sigma_end),
                            args.model.num_classes
                        )
                    )
                ).float().to(args.device) # Shape: (num_classes,)
        sigmas_np = sigmas.cpu().numpy()

        # Model Configuration
        used_activation = get_act(args.model.activation)
        scorenet = SimpleNet1d(data_dim=2, hidden_dim=args.model.hidden_dim, sigmas=sigmas, act=used_activation).to(args.device)
        scorenet.load_state_dict(torch.load(args.training.model_load_path, weights_only=True), strict=True)
        logging.info("Model loaded from '{}'.".format(args.training.model_load_path))

        # Sampler Configuration
        k_p = args.sampling.k_p
        k_i = args.sampling.k_i
        k_d = args.sampling.k_d
        k_i_window_size = args.sampling.k_i_window_size
        k_i_decay = args.sampling.k_i_decay
        k_d_decay = args.sampling.k_d_decay

        n_steps_each = args.sampling.n_steps_each
        step_lr = args.sampling.step_lr
        verbose = args.sampling.verbose
        
        assert type(k_i_window_size)==int, "`k_i_window_size` should be an integer."
        assert 0<k_i_window_size<=n_steps_each, "`k_i_window_size` should be in the range (0, n_steps_each]."


        # ======================================== PIDLD Sampling Start ========================================
        gen = torch.Generator()
        gen.manual_seed(42) # Set the seed for random initial noise, so that it will be the same across different runs.
        initial_noise = (16*torch.rand(args.data.n_test_samples,2,generator=gen)-8).to('cpu') # uniformly sampled from [-8, 8]
        samples_t = initial_noise.to(args.device)
        all_generated_samples = [samples_t.to('cpu')]

        with torch.no_grad(): # IMPORTANT!!!: disable gradient tracking during sampling.
            for c, sigma in enumerate(sigmas_np): # Iterate over noise levels
                labels = c * torch.ones(samples_t.shape[0], device=samples_t.device) # Get the noise labels for each sample
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas_np[-1]) ** 2
    
                if verbose:
                    logging.info("level: {:>4}, sigma: {:>7.3f}, k_p: {:>7.3f}, k_i: {:>7.3f}, k_d: {:>7.3f}".format(c, sigma, k_p, k_i, k_d))
                
                e_int = torch.zeros_like(samples_t).to(samples_t.device)
                e_prev = torch.zeros_like(samples_t).to(samples_t.device)
                e_diff = torch.zeros_like(samples_t).to(samples_t.device)
                e_t = torch.zeros_like(samples_t).to(samples_t.device)
                grads = deque(maxlen=k_i_window_size) # Deque: first in, first out.
    
                for t in range(n_steps_each): # Iterate over steps within each noise level
                    
                    # Update samples
                    grad = scorenet(samples_t, labels) # Compute gradient given samples and noise labels. Shape: (n_samples, *sample_shape)
                    grads.append(grad)
                    
                    e_int = sum(grads) / k_i_window_size # Use a moving window to sum the historical gradients
                    
                    e_prev = e_t # Update e_prev to e_t of the previous step
                    e_t = grad
                    e_diff = e_t - e_prev if t > 0 else torch.zeros_like(samples_t).to(samples_t.device) # For the first sampling step of a noise level, e_diff is set to 0 to avoid being too large
                    
                    # !IMPORTANT: Sample Updating Formula
                    noise = torch.randn_like(samples_t) # (n_samples, *sample_shape)
                    samples_t = samples_t + step_size * (k_p * grad + k_i * e_int + k_d * e_diff) + noise * np.sqrt(step_size * 2)

                    # Store intermediate samples
                    all_generated_samples.append(samples_t.to('cpu'))

                k_i = k_i * k_i_decay
                k_d = k_d * k_d_decay
    
            # Additional denoising step
            last_noise = (len(sigmas_np) - 1) * torch.ones(samples_t.shape[0], device=samples_t.device)
            last_noise = last_noise.long()
            samples_t = samples_t + sigmas_np[-1] ** 2 * scorenet(samples_t, last_noise)
            all_generated_samples.append(samples_t.to('cpu'))

        all_generated_samples = np.array([tensor.cpu().detach().numpy() for tensor in all_generated_samples]) # (num_classes * n_steps_each + 2, n_test_samples, 2)
        logging.info("Generated samples shape: {}".format(all_generated_samples.shape))
        # ======================================== PIDLD Sampling End ========================================


        # Evaluation
        frame_indices = np.linspace(1, len(all_generated_samples)-1, args.sampling.n_frames_each * args.model.num_classes + 1, dtype=int)

        logging.info("Start Evaluation...")
        kl_divergences, weights_preds, mu_preds, cov_preds = [], [], [], []

        weights_true = np.array(args.data.weights_true)
        mu_true = np.array(args.data.mu_true)
        cov_true = np.array(args.data.cov_true)
        for t in tqdm.tqdm(frame_indices, desc='Evaluating...'):
            weights_pred, mu_pred, cov_pred = gmm_estimation(all_generated_samples[t], n_components=2)
            kl = gmm_kl(weights_true, mu_true, cov_true, weights_pred, mu_pred, cov_pred, n_samples=100000)
            kl_divergences.append(kl)
            mu_preds.append(mu_pred)
            cov_preds.append(cov_pred)
            weights_preds.append(weights_pred)

        kl_divergence_final = kl_divergences[-1]

        # Result Saving
        logging.info("Saving Experiment Result...")
        result_dict = {
            'experiment_name': args.saving.experiment_name,
            'comment': args.saving.comment,
            'time_string': time_string,

            # Final metric
            'kl_divergence_final': kl_divergence_final,

            # Parameters of the sampling process
            'weights_preds_final': weights_preds[-1].tolist(),
            'mu_preds_final': mu_preds[-1].tolist(),
            'cov_preds_final': cov_preds[-1].tolist(),
        }
        result_save_path = os.path.join(experiment_dir, 'result.json')
        json.dump(result_dict, open(result_save_path, 'w'), indent=4)
        logging.info("Experiment result saved to '{}'.".format(result_save_path))

        record_dict = {
            # Metrics of each recorded frame of the sampling process
            'kl_divergences': np.array(kl_divergences),

            # Parameters of each recorded frame of the sampling process
            'weights_preds': np.array(weights_preds),
            'mu_preds': np.array(mu_preds),
            'cov_preds': np.array(cov_preds),
        }
        record_save_path = os.path.join(experiment_dir, 'record.npz')
        np.savez(record_save_path, **record_dict)
        logging.info("Experiment data record saved to '{}'.".format(record_save_path))

        # Ending Matters
        close_logger(logger)
        logging.info("Experiment finished.")
        return 0

    except Exception as e:
        logging.error(traceback.format_exc())
        close_logger(logger)
        return 1


if __name__ == '__main__':
    with open(os.path.join('configs', 'point_light.yml'), 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)
    args = dict2namespace(config_dict)
    main(args)
