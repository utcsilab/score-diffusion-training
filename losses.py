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

    return loss.mean(dim=0), nrmse.mean(dim=0), nrmse_img.mean(dim=0), torch.tensor(0), torch.tensor(0)

# No added noise SURE loss
def vanilla_sure_loss(scorenet, config):
    y = torch.tensor(config.current_sample[config.training.X_train])
    x = config.current_sample[config.training.X_label]
    sigma_w = config.current_sample['sigma_w']
    
    # Forward pass
    labels = torch.randint(0, len(scorenet.module.sigmas), (y.shape[0],), device=y.device)
    scorenet.module.sigmas = torch.ones(config.training.sigmas.shape).cuda()
    scorenet.module.logit_transform = True
    out = scorenet(y, labels)
    
    ## Measurement part of SURE
    meas_loss = torch.mean(torch.square(torch.abs(out - y)), dim=(-1, -2, -3))
    
    ## Divergence part of SURE
    # Sample random direction and increment
    random_dir = torch.randn_like(y)
    
    # Get model output in the scaled, perturbed directions
    out_eps = scorenet(y + config.optim.eps * random_dir, labels)
    
    # Normalized difference
    norm_diff = (out_eps - out) / config.optim.eps
    # Inner product with the direction vector
    div_loss = torch.mean(random_dir * norm_diff, dim=(-1, -2, -3))

    # Scale divergence loss
    div_loss = 2 * (torch.square(sigma_w)) * div_loss
          
    # Peek at true denoising loss
    with torch.no_grad():
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(out - x)), dim=(-1, -2, -3))) / (x.shape[-1] * x.shape[-2])
    
    return torch.mean(meas_loss + div_loss), torch.mean(meas_loss), torch.mean(denoising_loss), torch.tensor(0), torch.tensor(0)

def single_network_sure(scorenet, config):
    y = config.current_sample[config.training.X_train]
    x = config.current_sample[config.training.X_label]
    sigma_w = config.current_sample['sigma_w']

    # SURE Denoiser Forward pass
    labels = torch.randint(0, len(config.training.sigmas), (y.shape[0],), device=y.device)
    scorenet.module.sigmas[:] = sigma_w[0]
    denoiser_out = y + (scorenet(y, labels) * (scorenet.module.sigmas[0] ** 2))

    ## Measurement part of SURE
    meas_loss = torch.mean(torch.square(torch.abs(denoiser_out - y)), dim=(-1, -2, -3))
    
    ## Divergence part of SURE
    random_dir = torch.randn_like(y)
    
    # Get model output in the scaled, perturbed directions
    scorenet.module.sigmas[:] = sigma_w[0]+(sigma_w[0]*config.optim.eps)
    denoiser_out_eps = (y + config.optim.eps * random_dir) + (scorenet(y + config.optim.eps * random_dir, labels) *  (scorenet.module.sigmas[0] ** 2))
    
    # Normalized difference
    norm_diff = (denoiser_out_eps - denoiser_out) / config.optim.eps
    
    # Inner product with the direction vector
    div_loss = torch.mean(random_dir * norm_diff, dim=(-1, -2, -3))
    
    # Scale divergence loss
    div_loss = 2 * (torch.square(sigma_w)) * div_loss

    # Score Loss
    scorenet.module.sigmas = config.training.sigmas.clone().detach()
    used_sigmas = config.training.sigmas[labels].view(denoiser_out.shape[0], * ([1] * len(denoiser_out.shape[1:])))
    noise       = torch.randn_like(denoiser_out) * used_sigmas

    perturbed_samples = denoiser_out + noise

    # Desired output
    target = - 1 / (used_sigmas ** 2) * noise

    # Actual output
    scores = scorenet(perturbed_samples, labels)
    
    samples_est = perturbed_samples + (scores * (used_sigmas ** 2))
    samples_flatten = x.view(denoiser_out.shape[0], -1)
    samples_est_flatten = samples_est.view(samples_est.shape[0], -1)

    # L2 regression
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    
    # Multiply each sample by its weight
    score_loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** config.training.anneal_power
    
    # Loss weighting
    score_wt = config.training.score_wt * config.epoch
    loss = meas_loss + div_loss + (score_wt * score_loss)
    
    nrmse_img = (torch.norm((samples_flatten - samples_est_flatten), dim=1) / torch.norm(samples_flatten, dim=1))
    denoising_nrmse = torch.mean(torch.sum(torch.square(torch.abs(denoiser_out - x)), dim=(-1, -2, -3))) / (x.shape[-1] * x.shape[-2])

    return torch.mean(loss), torch.mean(denoising_nrmse), torch.mean(nrmse_img), torch.mean(score_loss), torch.mean(div_loss)

# No added noise SURE loss
def gsure_loss(scorenet, config):
    y = torch.tensor(config.current_sample[config.training.X_train])
    u = torch.tensor(config.current_sample['u'])
    x = config.current_sample[config.training.X_label]
    sigma_w = config.current_sample['sigma_w']
    
    # Forward pass
    labels = torch.randint(0, len(scorenet.module.sigmas), (y.shape[0],), device=y.device)
    scorenet.module.sigmas = torch.ones(config.training.sigmas.shape).cuda()
    scorenet.module.logit_transform = True
    out = scorenet(u, labels)
    
    ## Measurement part of SURE
    meas_loss = torch.mean(torch.square(torch.abs(out)), dim=(-1, -2, -3))
    
    ## Divergence part of SURE
    # Sample random direction and increment
    random_dir = torch.randn_like(u)
    
    # Get model output in the scaled, perturbed directions
    out_eps = scorenet(u + config.optim.eps * random_dir, labels)
    
    # Normalized difference
    norm_diff = (out_eps - out) / config.optim.eps
    # Inner product with the direction vector
    div_loss = torch.mean(random_dir * norm_diff, dim=(-1, -2, -3))

    # Scale divergence loss
    naive_mult = torch.sum(out * y, dim=(-1, -2, -3))
          
    # Peek at true denoising loss
    with torch.no_grad():
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(out - x)), dim=(-1, -2, -3))) / (x.shape[-1] * x.shape[-2])
    
    return torch.mean(meas_loss + 2 * (sigma_w**2 * div_loss - naive_mult)), torch.mean(meas_loss), torch.mean(denoising_loss), torch.tensor(0), torch.tensor(0)