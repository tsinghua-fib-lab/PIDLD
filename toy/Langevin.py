import logging
from typing import List, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def PID_ALD(x_mod: torch.Tensor, scorenet: nn.Module, sigmas: np.ndarray,
                n_steps_each: int=200, step_lr: float=0.000008,
                k_p: float=1.0, k_i: float=0.0, k_d: float=0.0,
                k_i_window_size: int=150, k_i_decay: float=1.00, k_d_decay: float=1.00,
                final_only: bool=False, denoise: bool=True, verbose: bool=False,
                log_freq: int=1,) -> Tuple[List[torch.Tensor], dict]:
    """
    Return the denoised samples using PID-controlled anneal Langevin Dynamics.
    For original anneal Langevin dynamics (without PID), please refer to https://github.com/ermongroup/ncsnv2/blob/master/models/__init__.py, line 20.

    Args:
        x_mod (torch.Tensor): Input (noisy) data. Shape: (batch_size, *sample_shape).
        scorenet (nn.Module): Score network to compute the gradient.
        sigmas (np.ndarray): Noise scales. Shape: (num_classes,).
        n_steps_each (int): Number of steps to take for each noise level.
        step_lr (float): Step size for each step.
        k_p (float): Proportional gain for the feedback control.
        k_i (float): Integral gain for the feedback control.
        k_d (float): Derivative gain for the feedback control.
        k_i_window_size (int): Moving window size for the integral gain.
        k_i_decay (float): Decay factor for the integral gain.
        k_d_decay (float): Decay factor for the derivative gain.
        final_only (bool): If True, only return the final denoised image.
        denoise (bool): If True, add an additional step to denoise the final image.
        verbose (bool): If True, print sampling information every `log_freq` steps.
        log_freq (int): Interval of logging.
    
    Returns:
        out (2-tuple of (list of Tensors) and dict):
            - out[0]:
                - if final_only is True: Final denoised samples. Length: 1.
                - if final_only is False: Denoised samples at each step. Length:
                    - num_classes * n_steps_each + 1 (if denoise=True).
                    - num_classes * n_steps_each (if denoise=False).
            - out[1]: A dictionary of norms and other metrics in the sampling process.
    """
    assert type(k_i_window_size)==int, "`k_i_window_size` should be an integer"
    assert 0<k_i_window_size<=n_steps_each, "`k_i_window_size` should be in the range (0, n_steps_each]"
    assert type(sigmas)== np.ndarray, "`sigmas` should be a numpy array"

    if verbose:
        logging.info('-'*80)
        logging.info('PID ALD Hyperparameters')
        logging.info('x_mod.shape: {}.'.format(x_mod.shape))
        logging.info('sigmas: {}.'.format(sigmas.tolist()))
        logging.info('n_steps_each: {}.'.format(n_steps_each))
        logging.info('step_lr: {}.'.format(step_lr))
        logging.info('k_p: {}, k_i: {}, k_d: {}.'.format(k_p, k_i, k_d))
        logging.info('k_i_window_size: {}, k_i_decay: {}, k_d_decay: {}.'.format(k_i_window_size, k_i_decay, k_d_decay))
        logging.info('final_only: {}, denoise: {}, verbose: {}, log_freq: {}.'.format(final_only, denoise, verbose, log_freq))
        logging.info('-'*80)

    def get_norm(x: torch.Tensor) -> float:
        """Compute the average Frobenius norm over a batch of tensors."""
        return torch.norm(x.view(x.shape[0], -1), p='fro', dim=-1).mean().item()

    with torch.no_grad(): # IMPORTANT!!!: disable gradient tracking during sampling.
        images = []

        sampler_record_dict = {
            "grad_norms": [], "e_int_norms": [], "e_diff_norms": [],
            "P_term_norms": [], "I_term_norms": [], "D_term_norms": [],
            "PID_term_norms": [], "noise_term_norms": [], "delta_term_norms": [],
            "snrs": [],
            "image_norms": [],
        }

        for c, sigma in enumerate(sigmas): # Iterate over noise levels
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c # Get the noise labels for each sample
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            if verbose:
                logging.info("level: {:>4}, sigma: {:>7.3f}, k_p: {:>7.3f}, k_i: {:>7.3f}, k_d: {:>7.3f}".format(c, sigma, k_p, k_i, k_d))
            
            e_int=torch.zeros_like(x_mod).to(x_mod.device)
            e_prev=torch.zeros_like(x_mod).to(x_mod.device)
            e_diff=torch.zeros_like(x_mod).to(x_mod.device)
            e_t=torch.zeros_like(x_mod).to(x_mod.device)
            grads = deque(maxlen=k_i_window_size) # Deque: first in, first out.

            for t in range(n_steps_each): # Iterate over steps within each noise level
                
                # Update samples
                grad = scorenet(x_mod, labels) # Compute gradient given samples and noise labels. Shape: (n_samples, *sample_shape)
                grads.append(grad)
                
                e_int = sum(grads) / k_i_window_size # Use a moving window to sum the historical gradients
                
                e_prev=e_t # Update e_prev to e_t of the previous step
                e_t = grad
                e_diff = e_t - e_prev if t>0 else torch.zeros_like(x_mod).to(x_mod.device) # For the first sampling step of a noise level, e_diff is set to 0 to avoid being too large
                
                # !IMPORTANT: Sample Updating Formula
                noise = torch.randn_like(x_mod) # (n_samples, *sample_shape)
                x_mod = x_mod + step_size * (k_p * grad + k_i * e_int + k_d * e_diff) + noise * np.sqrt(step_size * 2)
                
                if not final_only:
                    images.append(x_mod.to('cpu'))
                
                # Compute the norms and record them
                image_norm = get_norm(x_mod)
                grad_norm = get_norm(grad)
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
                PID_term_norm = get_norm(PID_term)
                noise_term_norm = get_norm(noise_term)
                delta_term_norm = get_norm(delta_term)
                snr = PID_term_norm/noise_term_norm # Signal to Noise Ratio

                sampler_record_dict['grad_norms'].append(grad_norm)
                sampler_record_dict['e_int_norms'].append(e_int_norm)
                sampler_record_dict['e_diff_norms'].append(e_diff_norm)
                sampler_record_dict['P_term_norms'].append(P_term_norm)
                sampler_record_dict['I_term_norms'].append(I_term_norm)
                sampler_record_dict['D_term_norms'].append(D_term_norm)
                sampler_record_dict['PID_term_norms'].append(PID_term_norm)
                sampler_record_dict['noise_term_norms'].append(noise_term_norm)
                sampler_record_dict['delta_term_norms'].append(delta_term_norm)
                sampler_record_dict['snrs'].append(snr)
                sampler_record_dict['image_norms'].append(image_norm)

                if verbose:
                    if t % log_freq == 0:
                        message = "level: {:>4}, step: {:>4}, step_size: {:>12.8f}".format(c, t, step_size)
                        message += ", image_norm: {:>12.8f}, snr: {:>12.8f}".format(image_norm, snr)
                        message += ", grad_norm: {:>12.8f}, e_int_norm: {:>12.8f}, e_diff_norm: {:>12.8f}".format(
                            grad_norm, e_int_norm, e_diff_norm)
                        message += ", P_norm: {:>12.8f}, I_norm: {:>12.8f}, D_norm: {:>12.8f}".format(
                            P_term_norm, I_term_norm, D_term_norm)
                        message += ", PID_norm: {:>12.8f}, noise_term_norm: {:>12.8f}, delta_term_norm: {:>12.8f}".format(
                            PID_term_norm, noise_term_norm, delta_term_norm)
                        logging.info(message)

            k_i = k_i * k_i_decay
            k_d = k_d * k_d_decay

        if denoise: # Additional denoising step
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')], sampler_record_dict
        else:
            return images, sampler_record_dict


def anneal_dsm_score_estimation(scorenet: nn.Module, samples: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """
    Denoising score matching loss of NCSN model. Reference: https://github.com/ermongroup/ncsnv2/blob/master/losses/dsm.py, line 3.

    Args:
        scorenet (nn.Module): a score network that takes in a sample and a label and outputs a score.
        samples (torch.Tensor): a tensor of shape (n_samples, *sample_shape)
        sigmas (torch.Tensor): The standard deviations of different noise levels. Shape (num_classes,)
    
    Returns:
        loss (Tensor): The loss of NCSN model. Shape: torch.Size([])
    """
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device) # Shape: (n_samples,)
    used_sigmas = sigmas[labels] # Choose noise standard deviations for each sample in the batch. Shape: (n_samples,)
    used_sigmas = used_sigmas.view(samples.shape[0], *([1] * len(samples.shape[1:]))) # -> (n_samples, 1, 1, 1)
    noise = torch.randn_like(samples) * used_sigmas # Shape: (n_samples, n_channels, height, width)
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels) # Shape: (n_samples, n_channels, height, width)
    target = target.view(target.shape[0], -1) # -> (n_samples, n_channels * height * width)
    scores = scores.view(scores.shape[0], -1) # -> (n_samples, n_channels * height * width)

    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** 2 # We use $\lambda(\sigma)$ to balance the loss, and it is proved that $\lambda(\sigma)=\sigma^2$

    return loss.mean(dim=0)
