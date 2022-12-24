import torch

def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):
    # This always enters during training
    if labels is None:
        # Randomly sample sigma
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)

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

# Direct denoising loss
def denoising_loss(model, x_noisy, x_clean, labels):
    # Fetch sigma from model
    sigma = model.sigmas[labels]
    
    # Sample noise
    n = torch.randn_like(x_noisy) * sigma[:, None, None, None]
    perturbed_samples = x_noisy + n
    # Get predicted output
    out = model.scaled_neutral_forward(
        perturbed_samples, sigma[:, None, None, None])
    
    # Supervised denoising
    meas_loss = torch.mean(torch.square(torch.abs(out - x_clean)), dim=(-1, -2, -3))
    
    # No divergence required
    div_loss = 0.
    
    return meas_loss, div_loss, sigma

# SURE loss
def sure_loss(model, y, x, labels, scaling='sigma', sigma_w=0., eps=1e-3):
    # Fetch sigma from model
    sigma = model.sigmas[labels]
    
    # Sample noise
    n = torch.randn_like(y) * sigma[:, None, None, None]
    perturbed_samples = y + n
    
    # Get predicted output
    if scaling == 'neutral':
        fw_sigma = torch.tensor(0., device='cuda')
    elif scaling == 'sigma':
        fw_sigma = torch.sqrt(torch.square(sigma_w) + torch.square(sigma))
        fw_sigma = fw_sigma[:, None, None, None]
    # Forward pass
    out = model.neutral_forward(perturbed_samples)
    
    ## Measurement part of SURE
    meas_loss = torch.mean(
        torch.square(torch.abs(out - perturbed_samples)), dim=(-1, -2, -3))
    
    ## Divergence part of SURE
    # Sample random direction and increment
    random_dir = torch.randn_like(y)
    
    # Get model output in the scaled, perturbed directions
    out_eps = model.neutral_forward(perturbed_samples + eps * random_dir)
    
    # Normalized difference
    norm_diff = (out_eps - out) / eps
    # Inner product with the direction vector
    div_loss = torch.mean(random_dir * norm_diff, dim=(-1, -2, -3))
    
    # Scale divergence loss
    div_loss = 2 * (torch.square(sigma) + torch.square(sigma_w)) * div_loss
          
    # Peek at true denoising loss
    with torch.no_grad():
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(
           out - x)), dim=(-1, -2, -3)))
    
    return meas_loss, div_loss, denoising_loss, sigma

# No added noise SURE loss
def vanilla_sure_loss(model, y, x, sigma_w=0., eps=1e-3):
    # Forward pass
    out = model.neutral_forward(y)
    
    ## Measurement part of SURE
    meas_loss = torch.mean(
        torch.square(torch.abs(out - y)), dim=(-1, -2, -3))
    
    ## Divergence part of SURE
    # Sample random direction and increment
    random_dir = torch.randn_like(y)
    
    # Get model output in the scaled, perturbed directions
    out_eps = model.neutral_forward(
        y + eps * random_dir)
    
    # Normalized difference
    norm_diff = (out_eps - out) / eps
    # Inner product with the direction vector
    div_loss = torch.mean(random_dir * norm_diff, dim=(-1, -2, -3))
    
    # Scale divergence loss
    div_loss = 2 * (torch.square(sigma_w)) * div_loss
          
    # Peek at true denoising loss
    with torch.no_grad():
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(
           out - x)), dim=(-1, -2, -3)))
    
    return meas_loss, div_loss, denoising_loss, None

# GSURE loss
def gsure_loss(model, y, x, x_ls, P, ortho_P, labels, sigma_w=0.,
               eps=1e-2):
    # Fetch sigma from model
    sigma = model.sigmas[labels]
    
    # Sample noise
    n = torch.randn_like(y) * sigma[:, None, None, None]
    perturbed_samples = y + n
    
    # Get predicted output
    out = model.gsure_forward(
        perturbed_samples, P, sigma[:, None, None, None])
    
    ## Projection part of GSURE
    proj_loss = torch.mean(torch.abs(torch.square(
        torch.matmul(ortho_P, out))), dim=(-1, -2, -3))
    
    ## Divergence part of GSURE
    # Sample random direction and increment
    random_dir = torch.randn_like(x)
    
    # Get model output in the scaled, perturbed directions
    out_eps = model.gsure_forward(
        perturbed_samples + eps * random_dir, P, sigma[:, None, None, None])
    
    # Normalized difference
    norm_diff = (out_eps - out) / eps
    # Inner product with the direction vector and scale
    div_loss  = torch.mean(random_dir * norm_diff, dim=(-1, -2, -3))
    div_loss  = 2 * (torch.square(sigma) + torch.square(sigma_w)) * div_loss
          
    ## Inner product part of GSURE
    inner_loss = -2 * torch.mean(out * x_ls)
    
    # Peek at true denoising loss
    with torch.no_grad():
        denoising_loss = torch.mean(torch.sum(torch.square(torch.abs(
           out - x)), dim=(-1, -2, -3)))
    
    return proj_loss, div_loss, inner_loss, denoising_loss, sigma

# GSURE equivalent of DSM [Pascal, '21 and Vincent '11]
def gsure_dsm_loss(model, y, x, forward, transposed, normal,
                   A_inv_pascal, A_inv_herm_pascal,
                   ortho_proj, labels, sigma_w, eps):
    # Fetch noise levels
    sigma = model.sigmas[labels]
    
    # Sample noise in full-dimensional space
    n       = torch.randn_like(y) * sigma[:, None, None, None]
    n_cplx  = n[:, 0] + 1j * n[:, 1]
    
    # Add noise to (already transposed) measurements
    y_cplx  = y[:, 0] + 1j * y[:, 1]
    x_tilde = y_cplx + n_cplx
    
    # Covariance matrix of mixed noise
    S = torch.square(sigma)[:, None, None] * \
        torch.eye(forward.shape[-1]).cuda()[None, ...] + \
        torch.square(sigma_w)[:, None, None] * normal
    
    # Get model prediction at noisy input
    x_tilde_real = torch.stack(
        (torch.real(x_tilde), torch.imag(x_tilde)), dim=1)
    out = model.neutral_forward(x_tilde_real)
    out_cplx = out[:, 0] + 1j * out[:, 1]
    
    # Measurement loss part - with a bunch of projections [Pascal '21]
    meas_loss = torch.mean(torch.square(torch.abs(
        torch.matmul(A_inv_pascal,
                     torch.matmul(forward, out_cplx) - x_tilde))),
        dim=(-1, -2))
    
    # Sample random direction and increment
    random_dir = torch.randn_like(x_tilde_real)
    random_dir_cplx = random_dir[:, 0] + 1j * random_dir[:, 1]
    # Get model output in the scaled, perturbed directions
    out_eps = model.neutral_forward(x_tilde_real + eps * random_dir)
    out_eps_cplx = out_eps[:, 0] + 1j * out_eps[:, 1]
    
    # Normalized difference
    norm_diff = (out_eps_cplx - out_cplx) / eps
    # Multiply with projection and pseudo-inverse [Pascal '21]
    proj_diff = torch.matmul(
        torch.matmul(A_inv_herm_pascal, ortho_proj), norm_diff)
    
    # Inner product with noise (and with covariance)
    div_loss = 2 * torch.mean(torch.conj(proj_diff) * \
                              torch.matmul(S, random_dir_cplx),
                              dim=(-1, -2))
    # Complex-to-real divergence
    div_loss = torch.real(div_loss) + torch.imag(div_loss)
    
    return meas_loss, div_loss

# GSURE equivalent of DSM [Pascal, '21 and Vincent '11]
def gsure_prediction_loss(model, y, x, forward, transposed, normal,
                   A_inv_pascal, A_inv_herm_pascal,
                   ortho_proj, labels, sigma_w, eps):
    # Fetch noise levels
    sigma = model.sigmas[labels]
    
    # Sample noise in full-dimensional space
    n       = torch.randn_like(y) * sigma[:, None, None, None]
    n_cplx  = n[:, 0] + 1j * n[:, 1]
    
    # Add noise to (already transposed) measurements
    y_cplx  = y[:, 0] + 1j * y[:, 1]
    x_tilde = y_cplx + n_cplx
    
    # Covariance matrix of mixed noise
    S = torch.square(sigma)[:, None, None] * \
        torch.eye(forward.shape[-1]).cuda()[None, ...] + \
        torch.square(sigma_w)[:, None, None] * normal
    
    # Get model prediction at noisy input
    x_tilde_real = torch.stack(
        (torch.real(x_tilde), torch.imag(x_tilde)), dim=1)
    out = model.neutral_forward(x_tilde_real)
    out_cplx = out[:, 0] + 1j * out[:, 1]
    
    # Measurement loss part - with a bunch of projections [Pascal '21]
    meas_loss = torch.mean(torch.square(torch.abs(
        torch.matmul(forward, out_cplx) - x_tilde)),
        dim=(-1, -2))
    
    # Sample random direction and increment
    random_dir = torch.randn_like(x_tilde_real)
    random_dir_cplx = random_dir[:, 0] + 1j * random_dir[:, 1]
    # Get model output in the scaled, perturbed directions
    out_eps = model.neutral_forward(x_tilde_real + eps * random_dir)
    out_eps_cplx = out_eps[:, 0] + 1j * out_eps[:, 1]
    
    # Normalized difference
    norm_diff = (out_eps_cplx - out_cplx) / eps
    # Multiply with forward operator [Pascal '21]
    proj_diff = torch.matmul(forward, norm_diff)
    proj_diff_real = torch.stack(
        (torch.real(proj_diff), torch.imag(proj_diff)), dim=1)
        
    # Compute noise with covariance
    shifted_noise = torch.matmul(S, random_dir_cplx)
    shifted_noise_real = torch.stack(
        (torch.real(shifted_noise), torch.imag(shifted_noise)), dim=1)
    
    # Inner product with noise (and with covariance)
    div_loss = 2 * torch.mean(proj_diff_real * shifted_noise_real,
                              dim=(-1, -2, -3))
    
    return meas_loss, div_loss

def yonina_gsure_loss(model, y, A_transpose, A_normal, H, H_transpose, 
                      P_ortho, sigma_w, labels, eps=1e-2):
    # Form GSURE measurements
    y_cplx  = y[:, 0] + 1j * y[:, 1]
    y_gsure = torch.matmul(A_transpose, y_cplx)
    
    # Fetch noise levels
    sigma = model.sigmas[labels]
    
    # Compute the inverse of C
    inv_cov = torch.linalg.inv(
        torch.square(sigma)[:, None, None] * \
            torch.eye(H_transpose.shape[1]).cuda()[None, ...] + \
            torch.square(sigma_w)[:, None, None] * A_normal)
    
    # Sample noise and perturb
    n       = torch.randn_like(y_gsure) * sigma[:, None, None]
    x_tilde = y_gsure + n
        
    # Obtain u
    u = torch.matmul(torch.matmul(H_transpose, inv_cov), x_tilde)
    u_real = torch.stack(
        (torch.real(u), torch.imag(u)), dim=1)
    
    # Obtain the ML estimate (using pseudo-inverse)
    dagger  = torch.linalg.pinv(
        torch.matmul(torch.matmul(H_transpose, inv_cov), H))
    x_ml    = torch.matmul(dagger, torch.matmul(H_transpose,
        torch.matmul(inv_cov, x_tilde)))
    
    # Get prediction at u
    out = model.neutral_forward(u_real)
    out_cplx = out[:, 0] + 1j * out[:, 1]
    
    # Sample random direction and increment
    random_dir = torch.randn_like(u_real)
    # Get model output in the scaled, perturbed directions
    out_eps = model.neutral_forward(u_real + eps * random_dir)
    
    # Normalized difference
    norm_diff      = (out_eps - out) / eps
    norm_diff_cplx = norm_diff[:, 0] + 1j * norm_diff[:, 1]
    # Project onto the range of H
    norm_diff_proj = torch.matmul(P_ortho, norm_diff_cplx)
    norm_diff_real = torch.stack(
        (torch.real(norm_diff_proj), torch.imag(norm_diff_proj)), dim=1)
    
    # Inner product with the direction vector
    div_loss = torch.mean(random_dir * norm_diff_real, dim=(-1, -2, -3))
    
    # Scale divergence loss
    div_loss = 2 * div_loss
    
    # Measurement term
    meas_loss = torch.mean(torch.square(torch.abs(
        torch.matmul(P_ortho, out_cplx))), dim=(-1, -2))
    
    # Inner product term
    inner_loss = -2 * torch.real(
        torch.mean(torch.conj(out_cplx) * x_ml, dim=(-1, -2)))
    
    return meas_loss, div_loss, inner_loss

# Can we implement GSURE?
def yonina_basic_gsure_loss(model, y, H, H_transpose, 
                            P_ortho, sigma_w, 
                            labels, eps=1e-2):
    # Form GSURE measurements
    y_cplx  = y[:, 0] + 1j * y[:, 1]
    y_gsure = y_cplx
    
    # Fetch noise levels
    # sigma = model.sigmas[labels]
    sigma  = torch.tensor([0.]).cuda()
    
    # Compute the inverse of C
    inv_cov = 1 / torch.square(sigma_w)[:, None, None]
    
    # Sample noise and perturb
    n       = torch.randn_like(y_gsure) * sigma[:, None, None]
    x_tilde = y_gsure + n
        
    # Obtain u
    u = inv_cov * torch.matmul(H_transpose, x_tilde)
    u_real = torch.stack(
        (torch.real(u), torch.imag(u)), dim=1)
    
    # Obtain the ML estimate (using pseudo-inverse)
    dagger  = torch.linalg.pinv(torch.matmul(H_transpose, H))
    x_ml    = torch.matmul(dagger, torch.matmul(H_transpose, x_tilde))
    
    # Get prediction at u
    out = model.neutral_forward(u_real)
    out_cplx = out[:, 0] + 1j * out[:, 1]
    
    # Sample random direction and increment
    random_dir = torch.randn_like(u_real)
    # Get model output in the scaled, perturbed directions
    out_eps = model.neutral_forward(u_real + eps * random_dir)
    
    # Normalized difference
    norm_diff      = (out_eps - out) / eps
    norm_diff_cplx = norm_diff[:, 0] + 1j * norm_diff[:, 1]
    # Project onto the range of H
    norm_diff_proj = torch.matmul(P_ortho, norm_diff_cplx)
    norm_diff_real = torch.stack(
        (torch.real(norm_diff_proj), torch.imag(norm_diff_proj)), dim=1)
    
    # Inner product with the direction vector
    div_loss = torch.mean(random_dir * norm_diff_real, dim=(-1, -2, -3))
    
    # Scale divergence loss
    div_loss = 2 * div_loss
    
    # Measurement term
    meas_loss = torch.mean(torch.square(torch.abs(
        torch.matmul(P_ortho, out_cplx))), dim=(-1, -2))
    
    # Inner product term
    inner_loss = -2 * torch.real(
        torch.mean(torch.conj(out_cplx) * x_ml, dim=(-1, -2)))
    
    return meas_loss, div_loss, inner_loss

# Can we implement GSURE? 
def basic_gsure_loss(model, y_model, H, H_transpose,
                     x_ml, P_ortho, sigma_w, labels, eps=1e-3):
    # Form GSURE measurements
    y_gsure = y_model[:, 0] + 1j * y_model[:, 1]
    
    # Fetch noise levels
    sigma = model.sigmas[labels]
    
    # Sample noise and perturb
    n       = torch.randn_like(y_gsure) * sigma[:, None, None]
    x_tilde = y_gsure + n
    
    # Compute the inverse of C
    inv_cov = 1 / (torch.square(sigma_w)[:, None, None] + \
                   torch.square(sigma)[:, None, None])
        
    # Obtain u
    u = inv_cov * torch.matmul(H_transpose, x_tilde)
    u_real = torch.stack((torch.real(u), torch.imag(u)), dim=1)
    
    # Get prediction at u
    out = model.neutral_forward(u_real)
    out_cplx = out[:, 0] + 1j * out[:, 1]
    
    # Sample random direction and increment
    random_dir = torch.randn_like(u_real)
    random_dir_cplx = random_dir[:, 0] + 1j * random_dir[:, 1]
    # Get model output in the scaled, perturbed directions
    out_eps = model.neutral_forward(u_real + eps * random_dir)
    
    # Normalized difference
    norm_diff      = (out_eps - out) / eps
    norm_diff_cplx = norm_diff[:, 0] + 1j * norm_diff[:, 1]
    # Project onto the range of H
    norm_diff_proj = torch.matmul(P_ortho, norm_diff_cplx)
    
    # Inner product with the direction vector
    div_loss = torch.mean(torch.real(torch.conj(random_dir_cplx) * \
                          norm_diff_proj), dim=(-1, -2))
    
    # Scale divergence loss
    div_loss = 2 * div_loss
    
    # Measurement term
    meas_loss = torch.mean(
        torch.square(torch.abs(
            torch.matmul(P_ortho, out_cplx - x_ml))), dim=(-1, -2))
    
    return meas_loss, div_loss, sigma

# DSM [Vincent, 2011]
def denoising_score_matching_loss(model, y, x, A_transpose, A_normal,
                                  sigma_w, labels):
    # Fetch noise levels
    sigma = model.sigmas[labels]
    # Sample noise
    n       = torch.randn_like(x) * sigma[:, None, None, None]
    n_cplx  = n[:, 0] + 1j * n[:, 1] 
    
    # Add noise to transposed measurements
    y_cplx  = y[:, 0] + 1j * y[:, 1]
    x_tilde = torch.matmul(A_transpose, y_cplx) + n_cplx
    
    # Fetch inverse covariance matrix
    inv_cov = torch.linalg.inv(
        torch.square(sigma)[:, None, None] * \
            torch.eye(A_transpose.shape[1]).cuda()[None, ...] + \
            torch.square(sigma_w)[:, None, None] * A_normal)
        
    # Get target
    target = -torch.matmul(inv_cov, n_cplx)
        
    # Predict
    x_tilde_real = torch.stack(
        (torch.real(x_tilde), torch.imag(x_tilde)), dim=1)
    out = model(x_tilde_real, labels)
    out_cplx = out[:, 0] + 1j * out[:, 1]
    
    # Weighted l2-loss
    loss = 1/2. * torch.mean(torch.square(sigma) * \
             torch.sum(torch.square(torch.abs(out_cplx - target)), dim=(-1, -2)))
    
    return loss