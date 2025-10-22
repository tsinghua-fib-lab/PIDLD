import logging

import numpy as np
import torch

from utils import batch_forward
from utils.hooks import SavingHook, VisualizationHook, RecordingHook, EvaluationHook


@torch.no_grad()
def PID_ALD(x_mod, scorenet, sigmas, n_steps_each, step_lr,
                    k_p, k_i, k_d, k_i_decay, k_d_decay, 
                    device, batch_size,
                    saving_hook: SavingHook,
                    visualization_hook: VisualizationHook,
                    recording_hook: RecordingHook,
                    evaluation_hook: EvaluationHook,
                    verbose=True, denoise=True,
    ):
    """
    PID-controlled anneal lagevin dynamics. Sample saving, evaluation and recording can be customized by hooks.

    Args:
        x_mod (torch.Tensor): Initial samples of shape (n_samples, n_channels, height, width), typically (10000,3,32,32).
        scorenet (nn.Module): Score network, which takes a batch of samples of shape (batch_size, n_channels, height, width)
            and a batch of labels of shape (batch_size,), and outputs a batch of gradients
            of shape (batch_size, n_channels, height, width).
        sigmas (np.ndarray): 1-d numpy array of noise levels.
        n_steps_each (int): Number of steps for each noise level.
        step_lr (float): Step size constant.
        
        k_p (float): Coefficient for Proportional gain.
        k_i (float): Coefficient for Integral gain.
        k_d (float): Coefficient for Derivative gain.
        k_i_decay (float): Decay rate for integral gain. The integral gain is multiplied by `k_i_decay` every noise level.
        k_d_decay (float): Decay rate for derivative gain. The derivative gain is multiplied by `k_d_decay` every noise level.
        
        device (torch.device): Device to run the model.
        batch_size (int): Batch size for data loading in score network gradient computation.

        saving_hook (Callable): The hook for saving samples.
        visualization_hook (Callable): The hook for visualization.
        recording_hook (Callable): The hook for recording.
        evaluation_hook (Callable): The hook for evaluating generation quality.

        verbose (bool): Whether to print the logging information.
        denoise (bool): Whether to add an additional step to denoise the final sample.

    Returns:
        x_mod (torch.Tensor): The final image after sampling.
    """
    x_mod = x_mod.to(device)
    e_int=torch.zeros_like(x_mod).to(x_mod.device) # The mean of historical gradients
    e_prev=torch.zeros_like(x_mod).to(x_mod.device)
    e_diff=torch.zeros_like(x_mod).to(x_mod.device)
    e_t=torch.zeros_like(x_mod).to(x_mod.device)
    
    global_step = 0 # Total number of sampling steps
    for c, sigma in enumerate(sigmas): # Iterate over noise levels
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        if verbose:
            logging.info("level: {:>4}, k_p={:>7.4f}, k_i={:>7.4f}, k_d={:>7.4f}, sigma={:>7.4f}".format(c, k_p, k_i, k_d, sigma))

        for t in range(n_steps_each): # Iterate over steps within each noise level

            # Proportional gain
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            grad = batch_forward(scorenet, x_mod, labels, batch_size=batch_size)
            
            # Integral gain
            e_int = (e_int * global_step + grad) / (global_step + 1) # Update the mean of historical gradients
            
            # Derivative gain
            e_prev=e_t
            e_t = grad
            e_diff = e_t - e_prev
            
            # !IMPORTANT: Updating formula
            noise = torch.randn_like(x_mod) # (n_samples, *sample_shape)
            x_mod = x_mod + step_size * (k_p * grad + k_i * e_int + k_d * e_diff) + noise * np.sqrt(step_size * 2)
            
            context = {
                'x_mod': x_mod,
                'level': c, 'step': t, 'global_step': global_step,
                'grad': grad, 'e_int': e_int, 'e_diff': e_diff,
                'noise': noise,
                'step_size': step_size,
                'k_p': k_p, 'k_i': k_i, 'k_d': k_d,
                'end': False,
            }
            
            # Extra operations by hook function (logging, saving, visualization, evaluation)
            saving_hook(**context)
            recording_hook(**context)
            evaluation_hook(**context)
            visualization_hook(**context)

            k_i = k_i * k_i_decay
            k_d = k_d * k_d_decay
            global_step = global_step + 1

    # Final denoising step
    if denoise:
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * (len(sigmas)-1)
        labels = labels.long()
        grad = batch_forward(scorenet, x_mod, labels, batch_size=batch_size)
        x_mod = x_mod + sigmas[-1] ** 2 * grad

        context = {
                'x_mod': x_mod,
                'level': c, 'step': t, 'global_step': global_step,
                'grad': grad, 'e_int': e_int, 'e_diff': e_diff,
                'noise': noise,
                'step_size': step_size,
                'k_p': k_p, 'k_i': k_i, 'k_d': k_d,
                'end': True,
            }
        saving_hook(**context)
        recording_hook(**context)
        evaluation_hook(**context)
        visualization_hook(**context)
    
    return x_mod.to('cpu')
