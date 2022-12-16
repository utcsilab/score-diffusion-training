import numpy as np
import sys
import torch
from scipy.stats import norm

def pairwise_dist(config, dataset, tqdm):
    # Construct pairwise distances
    # Set to true to follow [Song '20] exactly
    dist_matrix   = torch.zeros((len(dataset), len(dataset))).to('cuda')
    flat_channels = torch.tensor(dataset.channels.reshape((len(dataset), -1))).to('cuda')

    for idx in tqdm(range(len(dataset))):
        dist_matrix[idx] = torch.linalg.norm(flat_channels[idx][None, :] - flat_channels, dim=-1)

    np.savetxt(sys.path[0] + '/parameters/' + config.data.file + '.txt', [torch.max(dist_matrix.cpu())])

def sigma_rate(dataset, tqdm):
    # Apply Song's Technique 2
    candidate_gamma = np.logspace(np.log10(0.9), np.log10(0.99999), 1000)
    gamma_criterion = np.zeros((len(candidate_gamma)))
    dataset_shape = np.prod(dataset[0]['X'].shape)

    for idx, gamma in enumerate(candidate_gamma):
        gamma_criterion[idx] = \
            norm.cdf(np.sqrt(2 * dataset_shape) * (gamma - 1) + 3*gamma) - \
            norm.cdf(np.sqrt(2 * dataset_shape) * (gamma - 1) - 3*gamma)
    
    best_idx = np.argmin(np.abs(gamma_criterion - 0.5))
    return candidate_gamma[best_idx]

def step_size(config):
    # Choose the step size (epsilon) according to [Song '20]
    candidate_steps = np.logspace(-13, -8, 1000)
    step_criterion  = np.zeros((len(candidate_steps)))
    gamma_rate      = 1 / config.model.sigma_rate
    for idx, step in enumerate(candidate_steps):
        step_criterion[idx] = (1 - step / config.model.sigma_end ** 2) \
            ** (2 * config.model.num_classes) * (gamma_rate ** 2 -
                2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
                    1 - step / config.model.sigma_end ** 2) ** 2)) + \
                2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
                    1 - step / config.model.sigma_end ** 2) ** 2)
    best_idx = np.argmin(np.abs(step_criterion - 1.))

    return candidate_steps[best_idx]

def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):
    # This always enters during training
    if labels is None:
        # Randomly sample sigma
        labels = torch.randint(0, len(sigmas), 
                    (samples.shape[0],), device=samples.device)

    used_sigmas = sigmas[labels].view(samples.shape[0], * ([1] * len(samples.shape[1:])))
    noise       = torch.randn_like(samples) * used_sigmas

    perturbed_samples = samples + noise

    # Desired output
    target = - 1 / (used_sigmas ** 2) * noise

    # Actual output
    scores = scorenet(perturbed_samples, labels)

    noise_est = -(scores * (used_sigmas ** 2))
    samples_est = perturbed_samples - noise_est

    samples_flatten = samples.view(samples.shape[0], -1)
    samples_est_flatten = samples_est.view(samples_est.shape[0], -1)

    # L2 regression
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)

    # Multiply each sample by its weight
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    nrmse = (torch.norm((target - scores), dim=1) / torch.norm(target, dim=1))
    nrmse_img = (torch.norm((samples_flatten - samples_est_flatten), dim=1) / torch.norm(samples_flatten, dim=1))

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0), nrmse.mean(dim=0), nrmse_img.mean(dim=0)