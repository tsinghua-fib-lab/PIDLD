import os
import copy

import numpy as np
import scipy.linalg as linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader


os.environ['TORCH_HOME'] = 'model_weights'


def sample_fid_and_is(gen_images, inception_v3_model, mu_real, sigma_real, device, batch_size=128, num_workers=0):
    """
    Jointly calculate Frechet Inception Distance (FID) and Inception Score (IS) of generated images.

    The weights of inception model is downloaded from 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'.
    For CIFAR10 dataset, the mean and covariance of real features are downloaded from 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz'.
    For CelebA dataset, the mean and covariance of real features are downloaded from 'https://drive.google.com/drive/folders/1217uhIvLg9ZrYNKOR3XTRFSurt4miQrd?usp=sharing', metrics/pytorch_fid.zip, celeba_test_fid_stats.npz.

    Note that the penultimate activation of inception model is of shape (N, 1008), which aligns with the implementation of ncsnv2.

    Args:
        gen_images (torch.Tensor): Generated images of shape (N, 3, H, W), typically (10000, 3, 32, 32). Pixel values should be in range [0, 1].
        inception_v3_model (nn.Module): Pretrained Inception v3 model. The last layer should be `torch.nn.Linear(2048, 1008)` with `bias=True` and name `fc`.
        mu_real (np.ndarray): Mean of real features. Shape: (d,).
        sigma_real (np.ndarray): Covariance of real features. Shape: (d, d).
        device (torch.device): Computation device.
        batch_size (int): Batch size for dataloading.
        num_workers (int): Number of workers for data loading.

    Returns:
        (fid, is_mean, is_std) (tuple): FID and IS values.
    """
    # !!IMPORTANT: Normalize to [-1, 1]
    gen_images = 2 * gen_images - 1

    # Save the weights and biases of the final layer for later use
    inception_v3_model = copy.deepcopy(inception_v3_model) # Make a copy to avoid modifying the original model
    fc_weight = (inception_v3_model.fc.weight.data).to(device) # Shape: (1008, 2048)
    fc_bias = (inception_v3_model.fc.bias.data).to(device) # Shape: (1008,)
    
    # Remove final layer of the model to get the penultimate layer's activations
    inception_v3_model.fc = nn.Identity()
    inception_v3_model.aux_logits = False # Disable auxiliary output
    inception_v3_model.eval()

    # Calculate FID
    gen_images_data_loader = DataLoader(gen_images, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    gen_features = []
    with torch.no_grad():
        for gen_images_batch in tqdm.tqdm(gen_images_data_loader, desc='Calculating FID'):
            gen_images_batch_resized = F.interpolate(gen_images_batch,
                                        size=299, mode='bilinear', align_corners=False
                                        ).to(device) # Shape: (batch_size, 3, 299, 299)
            gen_features.append(inception_v3_model(gen_images_batch_resized).cpu().numpy())
    gen_features = np.concatenate(gen_features, axis=0) # Shape: (N, 1008)
    
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False) + np.eye(gen_features.shape[-1]) * 1e-6
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    # Calculate Inception Score
    gen_features_data_loader = DataLoader(torch.from_numpy(gen_features).to(device), batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    probs = []
    with torch.no_grad():
        for gen_features_batch in tqdm.tqdm(gen_features_data_loader, desc='Calculating IS'):
            logits = F.linear(gen_features_batch, fc_weight, fc_bias) # Apply the final layer to the penultimate layer's activations
            preds = nn.functional.softmax(logits, dim=1)
            probs.append(preds.cpu().detach().numpy())
    probs = np.concatenate(probs, axis=0) # Shape: (N, 1008)

    scores = []
    n_splits = 10
    split_size = len(probs) // n_splits
    for i in range(n_splits): # Calculate score for each subset
        subset = probs[i*split_size : (i+1)*split_size] # Split data into chunks
        py = np.mean(subset, axis=0) # Shape: (d,); marginal probability p(y)
        kl = subset * (np.log(subset) - np.log(py))
        kl = np.sum(kl, axis=1)
        average_kl = np.mean(kl) # Scalar
        is_score = np.exp(average_kl) # Scalar
        scores.append(is_score)
    
    is_mean = np.mean(scores)
    is_std = np.std(scores)

    return fid, is_mean, is_std


# Source: https://github.com/ermongroup/ncsnv2/blob/1bcea03bd97d76a8357b9d674fa0bae5ff0f1093/evaluation/fid_score.py, line 124 - 178
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    #print(1, time.time())
    #print(sigma1.shape, sigma2.shape)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False) # Slow, requires 6s
    #print(2, time.time())
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
