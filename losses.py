import torch
import numpy as np

def anneal_dsm_score_estimation(scorenet, config):
    # This always enters during training
    samples = config.current_sample[config.training.X_train]
    labels = torch.randint(0, len(config.training.sigmas), (samples.shape[0],), device=samples.device)

    used_sigmas = config.training.sigmas[labels].view(samples.shape[0], * ([1] * len(samples.shape[1:])))
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
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** config.training.anneal_power

    nrmse = (torch.norm((target - scores), dim=1) / torch.norm(target, dim=1))
    nrmse_img = (torch.norm((samples_flatten - samples_est_flatten), dim=1) / torch.norm(samples_flatten, dim=1))

    return loss.mean(dim=0), nrmse.mean(dim=0), nrmse_img.mean(dim=0)

# No added noise SURE loss
def vanilla_sure_loss(scorenet, config):
    y = torch.tensor(config.current_sample[config.training.X_train])
    x = config.current_sample[config.training.X_label]
    sigma_w = config.current_sample['sigma_w']
    
    # Forward pass
    labels = torch.randint(0, len(scorenet.sigmas), (y.shape[0],), device=y.device)
    scorenet.sigmas = torch.ones(config.training.sigmas.shape).cuda()
    scorenet.logit_transform = True
    out = scorenet.forward(y, labels)
    
    ## Measurement part of SURE
    meas_loss = torch.mean(torch.square(torch.abs(out - y)), dim=(-1, -2, -3))
    
    ## Divergence part of SURE
    # Sample random direction and increment
    random_dir = torch.randn_like(y)
    
    # Get model output in the scaled, perturbed directions
    out_eps = scorenet.forward(y + config.optim.eps * random_dir, labels)
    
    # Normalized difference
    norm_diff = (out_eps - out) / config.optim.eps
    # Inner product with the direction vector
    div_loss = torch.mean(random_dir * norm_diff, dim=(-1, -2, -3))

    # Scale divergence loss
    div_loss = 2 * (torch.square(sigma_w)) * div_loss
          
    # Peek at true denoising loss
    with torch.no_grad():
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(out - x)), dim=(-1, -2, -3))) / (x.shape[-1] * x.shape[-2])
    
    return torch.mean(meas_loss + div_loss), torch.mean(meas_loss), torch.mean(denoising_loss)

def single_level_sure(scorenet, config):
    y = config.current_sample[config.training.X_train]
    x = config.current_sample[config.training.X_label]
    sigma_w = config.current_sample['sigma_w']

    # SURE Denoiser Forward pass
    labels = torch.randint(0, len(config.denoiser.sigmas), (y.shape[0],), device=y.device)
    denoiser_out = config.denoiser.forward(y, labels)

    ## Measurement part of SURE
    meas_loss = torch.mean(torch.square(torch.abs(denoiser_out - y)), dim=(-1, -2, -3))
    
    ## Divergence part of SURE
    # Sample random direction and increment
    random_dir = torch.randn_like(y)
    
    # Get model output in the scaled, perturbed directions
    denoiser_out_eps = config.denoiser.forward(y + config.optim.eps * random_dir, labels)
    
    # Normalized difference
    norm_diff = (denoiser_out_eps - denoiser_out) / config.optim.eps
    
    # Inner product with the direction vector
    div_loss = torch.mean(random_dir * norm_diff, dim=(-1, -2, -3))
    
    # Scale divergence loss
    div_loss = 2 * (torch.square(sigma_w)) * div_loss

    # # Peek at true denoising loss
    with torch.no_grad():
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(denoiser_out - x)), dim=(-1, -2, -3))) / (x.shape[-1] * x.shape[-2])
    
    denoiser_out = (denoiser_out * config.data.std) + (config.data.mean)

    # Score Loss
    used_sigmas = config.training.sigmas[labels].view(denoiser_out.shape[0], * ([1] * len(denoiser_out.shape[1:])))
    noise       = torch.randn_like(denoiser_out) * used_sigmas

    perturbed_samples = denoiser_out + noise

    # Desired output
    target = - 1 / (used_sigmas ** 2) * noise

    # Actual output
    scores = scorenet(perturbed_samples, labels)

    noise_est = -(scores * (used_sigmas ** 2))
    samples_est = perturbed_samples - noise_est

    samples_flatten = x.view(denoiser_out.shape[0], -1)
    samples_est_flatten = samples_est.view(samples_est.shape[0], -1)

    # L2 regression
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    
    # Multiply each sample by its weight
    score_loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** config.training.anneal_power
    
    # Loss weighting
    loss = (config.training.sure_wt[config.epoch] * (meas_loss + div_loss)) + (config.training.score_wt[config.epoch] * score_loss)
    
    nrmse_img = (torch.norm((samples_flatten - samples_est_flatten), dim=1) / torch.norm(samples_flatten, dim=1))

    return torch.mean(loss), torch.mean(score_loss), torch.mean(nrmse_img)

# Multi-Level SURE loss
def multi_level_sure(scorenet, config):
    y = config.current_sample[config.training.X_train]
    x = config.current_sample[config.training.X_label]
    sigma_w = config.current_sample['sigma_w']
    
    # Forward pass
    labels = torch.randint(0, len(scorenet.sigmas), (y.shape[0],), device=y.device)
    
    # Fetch sigma from model
    sigma = config.training.sigmas[labels]
    scorenet.sigmas = torch.ones(config.training.sigmas.shape).cuda()
    scorenet.logit_transform = True
    
    # Sample noise
    n = torch.randn_like(y) * sigma[:, None, None, None]
    perturbed_samples = y + n
    
    # Get predicted output
    if config.training.scaling == 'neutral':
        fw_sigma = torch.tensor(0., device='cuda')
    elif config.training.scaling == 'sigma':
        fw_sigma = torch.sqrt(torch.square(sigma_w) + torch.square(sigma))
        fw_sigma = fw_sigma[:, None, None, None]
    
    # Forward pass
    out = scorenet.forward(perturbed_samples, labels)
    
    ## Measurement part of SURE
    meas_loss = torch.mean(torch.square(torch.abs(out - perturbed_samples)), dim=(-1, -2, -3))
    
    ## Divergence part of SURE
    # Sample random direction and increment
    random_dir = torch.randn_like(y)
    
    # Get model output in the scaled, perturbed directions
    out_eps = scorenet.forward(perturbed_samples + config.optim.eps * random_dir, labels)
    
    # Normalized difference
    norm_diff = (out_eps - out) / config.optim.eps
    
    # Inner product with the direction vector
    div_loss = torch.mean(random_dir * norm_diff, dim=(-1, -2, -3))
    
    # Scale divergence loss
    div_loss = 2 * (torch.square(sigma) + torch.square(sigma_w)) * div_loss
          
    # Peek at true denoising loss
    with torch.no_grad():
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(out - x)), dim=(-1, -2, -3))) / (x.shape[-1] * x.shape[-2])
    
    # Scale noise level with function lambda (sigma)
    meas_loss = torch.pow(torch.sqrt(torch.square(sigma) + torch.square(sigma_w)), -2) * meas_loss
    div_loss  = torch.pow(torch.sqrt(torch.square(sigma) + torch.square(sigma_w)), -2) * div_loss

    return torch.mean(meas_loss + div_loss), torch.mean(meas_loss), torch.mean(denoising_loss)