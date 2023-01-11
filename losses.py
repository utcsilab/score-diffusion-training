import torch

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

# SURE loss
def sure_loss(scorenet, config):
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
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(out - x)), dim=(-1, -2, -3)))
    
    # Scale noise level with function lambda (sigma)
    meas_loss = torch.pow(torch.sqrt(torch.square(sigma) + torch.square(sigma_w)), -2) * meas_loss
    div_loss  = torch.pow(torch.sqrt(torch.square(sigma) + torch.square(sigma_w)), -2) * div_loss

    return torch.mean(meas_loss + div_loss), torch.mean(meas_loss), torch.mean(denoising_loss)

# No added noise SURE loss
def vanilla_sure_loss(scorenet, config):
    y = torch.tensor(config.current_sample[config.training.X_train], requires_grad=True)
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
    
    # out.sum().backward()
    # div_loss = torch.mean(y.grad)
    
    # Scale divergence loss
    div_loss = 2 * (torch.square(sigma_w)) * div_loss
          
    # Peek at true denoising loss
    with torch.no_grad():
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(
           out - x)), dim=(-1, -2, -3)))
    
    return torch.mean(meas_loss + div_loss), torch.mean(meas_loss), torch.mean(denoising_loss)

# GSURE loss
def gsure_loss(scorenet, config):
    y = config.current_sample[config.training.X_train]
    x = config.current_sample[config.training.X_label]
    x_ls = config.current_sample['x_ls']
    P = config.current_sample['P']
    ortho_P = config.current_sample['P_ortho']
    sigma_w = config.current_sample['sigma_w']
    
    # Fetch sigma from model
    labels = torch.randint(0, len(scorenet.sigmas), (y.shape[0],), device=y.device)
    sigma = config.training.sigmas[labels]
    scorenet.sigmas = torch.ones(config.training.sigmas.shape).cuda()
    scorenet.logit_transform = True
    
    # Sample noise
    n = torch.randn_like(y) * sigma[:, None, None, None]
    perturbed_samples = y + n
    
    # Get predicted output
    # Apply adjoint (complex-valued)
    y_cplx = perturbed_samples[:, 0] + 1j * perturbed_samples[:, 1]
    h_cplx = torch.matmul(P.conj().transpose(-1, -2), y_cplx)
    h      = torch.stack((torch.real(h_cplx), torch.imag(h_cplx)), dim=1)

    out = scorenet.forward(h, labels)
    
    ## Projection part of GSURE
    proj_loss = torch.mean(torch.abs(torch.square(torch.matmul(ortho_P, out))), dim=(-1, -2, -3))
    
    ## Divergence part of GSURE
    # Sample random direction and increment
    random_dir = torch.randn_like(x)
    
    # Get model output in the scaled, perturbed directions
    y_out = perturbed_samples + config.optim.eps * random_dir
    y_cplx = y_out[:, 0] + 1j * y_out[:, 1]
    h_cplx = torch.matmul(P.conj().transpose(-1, -2), y_cplx)
    h      = torch.stack((torch.real(h_cplx), torch.imag(h_cplx)), dim=1)

    out_eps = scorenet.forward(h, labels)
    
    # Normalized difference
    norm_diff = (out_eps - out) / config.optim.eps
    # Inner product with the direction vector and scale
    div_loss  = torch.mean(random_dir * norm_diff, dim=(-1, -2, -3))
    div_loss  = 2 * (torch.square(sigma) + torch.square(sigma_w)) * div_loss
          
    ## Inner product part of GSURE
    inner_loss = -2 * torch.mean(out * x_ls)
    
    # Peek at true denoising loss
    with torch.no_grad():
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(out - x)), dim=(-1, -2, -3)))
    
    return torch.mean(proj_loss + div_loss), torch.mean(proj_loss), torch.mean(denoising_loss)