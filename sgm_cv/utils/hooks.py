import os
import logging
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from utils import get_norm
from evaluation.metrics import sample_fid_and_is


class RecordingHook:
    """A hook to manage sampler recording and logging."""
    def __init__(self, verbose=True):
        self.sampler_record_dict = {
            'grad_norms': [], 'e_int_norms': [], 'e_diff_norms': [],
            'P_term_norms': [], 'I_term_norms': [], 'D_term_norms': [],
            'IP_ratios': [], 'DP_ratios': [],
            'PID_term_norms': [], 'noise_term_norms': [], 'delta_term_norms': [],
            'snrs': [],
            'image_norms': [],
        }
        self.verbose=verbose

    def __call__(self, **context):
        end = context['end']
        if end==True:
            return
        x_mod = context['x_mod']
        level, step, global_step = context['level'], context['step'], context['global_step']
        grad, e_int, e_diff = context['grad'], context['e_int'], context['e_diff']
        noise = context['noise']
        step_size = context['step_size']
        k_p, k_i, k_d = context['k_p'], context['k_i'], context['k_d']

        grad_norm = get_norm(grad)
        noise_norm = get_norm(noise)
        image_norm = get_norm(x_mod)
        e_int_norm = get_norm(e_int)
        e_diff_norm = get_norm(e_diff)

        P_term = step_size * k_p * grad
        I_term = step_size * k_i * e_int
        D_term = step_size * k_d * e_diff
        PID_term = P_term + I_term + D_term
        noise_term = noise * np.sqrt(step_size * 2)
        delta_term = PID_term + noise_term
                
        P_term_norm = get_norm(P_term)
        I_term_norm = get_norm(I_term)
        D_term_norm = get_norm(D_term)
        IP_ratio = I_term_norm/P_term_norm
        DP_ratio = D_term_norm/P_term_norm
        PID_term_norm = get_norm(PID_term)
        noise_term_norm = get_norm(noise_term)
        delta_term_norm = get_norm(delta_term)
        snr = PID_term_norm/noise_term_norm # Signal to Noise Ratio

        # Recording
        self.sampler_record_dict['grad_norms'].append(grad_norm)
        self.sampler_record_dict['e_int_norms'].append(e_int_norm)
        self.sampler_record_dict['e_diff_norms'].append(e_diff_norm)
        self.sampler_record_dict['P_term_norms'].append(P_term_norm)
        self.sampler_record_dict['I_term_norms'].append(I_term_norm)
        self.sampler_record_dict['D_term_norms'].append(D_term_norm)
        self.sampler_record_dict['IP_ratios'].append(IP_ratio)
        self.sampler_record_dict['DP_ratios'].append(DP_ratio)
        self.sampler_record_dict['PID_term_norms'].append(PID_term_norm)
        self.sampler_record_dict['noise_term_norms'].append(noise_term_norm)
        self.sampler_record_dict['delta_term_norms'].append(delta_term_norm)
        self.sampler_record_dict['snrs'].append(snr)
        self.sampler_record_dict['image_norms'].append(image_norm)

        if self.verbose:
            message = "level: {:>4}, step: {:>4}".format(level, step)
            message += ", image_norm: {:>13.8f}, step_size: {:>13.8f}".format(image_norm, step_size)
            message += ", grad_norm: {:>13.8f}, noise_norm: {:>13.8f}, snr: {:>13.8f}".format(
                grad_norm, noise_norm, snr)
            message += ", e_int_norm: {:>13.8f}, e_diff_norm: {:>13.8f}".format(
                e_int_norm, e_diff_norm)
            message += ", P_norm: {:>13.8f}, I_norm: {:>13.8f}, D_norm: {:>13.8f}".format(
                P_term_norm, I_term_norm, D_term_norm)
            message += ", IP_ratio: {:>13.8f}, DP_ratio: {:>13.8f}".format(
                IP_ratio, DP_ratio)
            message += ", PID_norm: {:>13.8f}, noise_term_norm: {:>13.8f}, delta_term_norm: {:>13.8f}".format(
                PID_term_norm, noise_term_norm, delta_term_norm)
            logging.info(message)


class SavingHook:
    """A hook to manage sample saving."""
    def __init__(self, save, freq, last_only, sample_save_dir, verbose):
        self.save = save
        self.freq = freq
        self.last_only = last_only
        self.sample_save_dir = sample_save_dir
        self.verbose = verbose

    def __call__(self, **context):
        x_mod = context['x_mod']
        level, step, global_step = context['level'], context['step'], context['global_step']
        end = context['end']
        if self.save:
            sample_save_path = None
            if end==False and not self.last_only and global_step % self.freq == 0:
                sample_save_path = os.path.join(self.sample_save_dir, 'samples_level_{:03d}_step_{:03d}.pth'.format(level, step))
            elif end==True:
                sample_save_path = os.path.join(self.sample_save_dir, 'samples_final_denoised.pth')
            if sample_save_path is not None:
                torch.save(x_mod.detach().cpu(), sample_save_path)
                if self.verbose:
                    logging.info("level: {:>4}, step: {:>4}, Sample saved to '{}'".format(level, step, sample_save_path))


class VisualizationHook:
    """A hook to manage image grid visualization and saving."""
    def __init__(self, save, freq, last_only, nrow, sample_save_dir, verbose):
        self.save = save
        self.freq = freq
        self.last_only = last_only
        self.nrow = nrow
        self.sample_save_dir = sample_save_dir
        self.verbose = verbose

    def __call__(self, **context):
        x_mod = context['x_mod']
        level, step, global_step = context['level'], context['step'], context['global_step']
        end = context['end']
        if self.save:
            sample_save_path = None
            nrow=self.nrow
            if end==False and not self.last_only and global_step % self.freq == 0:
                sample_save_path = os.path.join(self.sample_save_dir, 'image_grid_{}x{}_level_{:03d}_step_{:03d}.png'.format(nrow, nrow, level, step))
            elif end==True:
                sample_save_path = os.path.join(self.sample_save_dir, 'image_grid_final_denoised.png')
            if sample_save_path is not None:
                plt.figure(figsize=(8,8))
                grid=torchvision.utils.make_grid(x_mod.detach().cpu()[:nrow*nrow], nrow=nrow, padding=2).permute(1,2,0).numpy()
                plt.imshow(grid)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(sample_save_path, bbox_inches='tight')
                plt.close()
                image_grid_save_path = sample_save_path.replace('.png', '.pth')
                torch.save(grid, image_grid_save_path)
                if self.verbose:
                    logging.info("level: {:>4}, step: {:>4}, figure saved to '{}'".format(level, step, sample_save_path))
                    logging.info("level: {:>4}, step: {:>4}, image grid tensor saved to '{}'".format(level, step, image_grid_save_path))


class EvaluationHook:
    """A hook to manage FID and IS evaluation."""
    def __init__(self, inception_v3_model, mu_real, sigma_real, device,
                    batch_size, num_workers,
                    evaluate, freq, last_only,
                    verbose=True
                ):
        self.evaluate = evaluate
        self.evaluate_func = partial(sample_fid_and_is, inception_v3_model=inception_v3_model,
                                        mu_real=mu_real, sigma_real=sigma_real,
                                        device=device, batch_size=batch_size, num_workers=num_workers)
        self.freq = freq
        self.last_only = last_only

        self.metric_record_dict = {
            'fids': [],
            'is_means': [],
            'is_stds': []
        }

        self.verbose = verbose
    
    def __call__(self, **context):
        x_mod = context['x_mod']
        level, step, global_step = context['level'], context['step'], context['global_step']
        end = context['end']
        flag=False
        if self.evaluate:
            if end==False and not self.last_only and global_step % self.freq == 0:
                flag=True
            elif end==True:
                flag=True
        if flag:
            fid, is_mean, is_std = self.evaluate_func(x_mod)
            self.metric_record_dict['fids'].append(fid)
            self.metric_record_dict['is_means'].append(is_mean)
            self.metric_record_dict['is_stds'].append(is_std)
            if self.verbose:
                logging.info("level: {:>4}, step: {:>4}, FID: {:>11.6f}, IS_mean: {:>11.6f}, IS_std: {:>11.6f}".format(
                    level, step, fid, is_mean, is_std))
