#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, sys, os, json, argparse
from dotmap import DotMap
sys.path.append('.')

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str)

from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

from ncsnv2.models        import get_sigmas
from ncsnv2.models.ema    import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest, NCSNv2Deeper, NCSNv2
from ncsnv2.losses        import get_optimizer

from parameters import *
from losses     import *

from loaders          import *
from torch.utils.data import DataLoader

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

# Model config
config = DotMap(json.load(open(parser.parse_args().config_path)))

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.model.gpu)
config.device                      = 'cuda:0'

# Inner model
config.model.ema           = True
config.model.ema_rate      = 0.999 # Exponential moving average, for stable FID scores (Song'20)
config.model.normalization = 'InstanceNorm++'
config.model.nonlinearity  = 'elu'
config.model.sigma_dist    = 'geometric'

# Optimizer
config.optim.weight_decay  = 0.000 # No weight decay
config.optim.optimizer     = 'Adam'
config.optim.lr            = 0.0001
config.optim.beta1         = 0.9
config.optim.amsgrad       = False
config.optim.eps           = 0.001

# Training
config.training.num_workers    = 8
config.training.anneal_power   = 2
config.training.log_all_sigmas = False
config.training.eval_freq      = 100 # In steps

# Data
config.data.channels       = 2 # {Re, Im}

print('\nDataset: ' + config.data.file)
print('Dataloader: ' + config.data.dataloader)
print('Loss Function: ' + config.model.loss + '\n')

# Get datasets and loaders for channels
dataset     = globals()[config.data.dataloader](config)
dataloader  = DataLoader(dataset, batch_size=config.training.batch_size, 
                         shuffle=True, num_workers=config.training.num_workers, 
                         drop_last=True)

pairwise_dist_path = './parameters/' + config.data.file + '.txt'
if not os.path.exists(pairwise_dist_path):
    pairwise_dist(config, dataset, tqdm)

config.data.image_size = [dataset.channels.shape[1], dataset.channels.shape[2]]
config.model.sigma_begin = np.loadtxt(pairwise_dist_path)

if isinstance(config.model.sigma_rate, str):
    config.model.sigma_rate = globals()[config.model.sigma_rate](dataset, config)

if not config.model.sigma_end:
      config.model.sigma_end = config.model.sigma_begin * config.model.sigma_rate ** (config.model.num_classes - 1)

config.model.step_size = step_size(config)

print('\nSigma Begin: ' + str(config.model.sigma_begin))
print('Sigma Rate: ' + str(config.model.sigma_rate))
print('Sigma End: ' + str(config.model.sigma_end) + '\n')

# Get a model
if config.model.depth == 'large':
    diffuser = NCSNv2Deepest(config)
elif config.model.depth == 'medium':
    diffuser = NCSNv2Deeper(config)
elif config.model.depth == 'low':
    diffuser = NCSNv2(config)

diffuser = diffuser.cuda()
# Get a collection of sigma values
if config.model.get_sigmas:
    sigmas = globals()[config.model.get_sigmas](config)
else:
    sigmas = get_sigmas(config)
    
diffuser.sigmas = sigmas

# Get optimizer
optimizer = get_optimizer(config, diffuser.parameters())

# Counter
start_epoch = 0
step = 0
if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(diffuser)

# More logging
config.log_path = './models/' + config.data.file + '_' + config.data.dataloader + '/\
sigma_begin%d_sigma_end%.4f_num_classes%.1f_sigma_rate%.4f_epochs%.1f' % (
    config.model.sigma_begin, config.model.sigma_end,
    config.model.num_classes, config.model.sigma_rate, 
    config.training.n_epochs)

if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)

# No sigma logging
hook = test_hook = None

# Logged metrics
print('\n')
train_loss, train_nrmse, train_nrmse_img  = [], [], []

for epoch in tqdm(range(start_epoch, config.training.n_epochs)):
    for i, sample in tqdm(enumerate(dataloader)):
        # Safety check
        diffuser.train()
        step += 1
        
        # Move data to device
        for key in sample:
            sample[key] = sample[key].cuda()

        # Get loss on Hermitian channels
        loss, nrmse, nrmse_img = globals()[config.model.loss](
            diffuser, sample[config.training.X_train], sigmas, None, 
            config.training.anneal_power, hook)
        
        # Keep a running loss
        if step == 1:
            running_loss = loss.item()
            running_nrmse = nrmse.item()
            running_nrmse_img = nrmse_img.item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.item()
            running_nrmse = 0.99 * running_nrmse + 0.01 * nrmse.item()
            running_nrmse_img = 0.99 * running_nrmse_img + 0.01 * nrmse_img.item()
    
        # Log
        train_loss.append(loss.item())
        train_nrmse.append(nrmse.item())
        train_nrmse_img.append(nrmse_img.item())
        
        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # EMA update
        if config.model.ema:
            ema_helper.update(diffuser)
            
        # Verbose
        if step % config.training.eval_freq == 0:
            # Print
            print('Epoch %d, Step %d, Loss (EMA) %.3f, NRMSE (Noise) %.3f, NRMSE (Image) %.3f' % 
                (epoch, step, running_loss, running_nrmse, running_nrmse_img))
        
# Save snapshot
torch.save({'diffuser': diffuser,
            'model_state': diffuser.state_dict(),
            'config': config,
            'loss': train_loss,
            'nrmse_noise': train_nrmse,
            'nrmse_img': train_nrmse_img}, 
   os.path.join(config.log_path, 'final_model.pt'))