#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, sys, os, argparse
sys.path.append('./')

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

from parameters import pairwise_dist
from parameters import sigma_rate
from parameters import step_size
from parameters import anneal_dsm_score_estimation

from loaders          import Knee_Basis_Loader
from torch.utils.data import DataLoader

from dotmap import DotMap

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--num_classes', type=int, default=2600)
parser.add_argument('--sigma_rate', type=float, default=0.9945)
parser.add_argument('--file', type=str, default='knee_basis1_norm95')
parser.add_argument('--image_size', nargs='+', type=int, default=[256, 256])
parser.add_argument('--depth', type=str, default='large')
args = parser.parse_args()

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Model config
config          = DotMap()
config.device   = 'cuda:0'
# Inner model
config.model.ema           = True
config.model.ema_rate      = 0.999 # Exponential moving average, for stable FID scores (Song'20)
config.model.normalization = 'InstanceNorm++'
config.model.nonlinearity  = 'elu'
config.model.sigma_dist    = 'geometric'
config.model.num_classes   =  int(args.num_classes) # Number of train sigmas and 'N'
config.model.ngf           = int(args.ngf)

# Optimizer
config.optim.weight_decay  = 0.000 # No weight decay
config.optim.optimizer     = 'Adam'
config.optim.lr            = 0.0001
config.optim.beta1         = 0.9
config.optim.amsgrad       = False
config.optim.eps           = 0.001

# Training
config.training.batch_size     = int(args.batch_size)
config.training.num_workers    = 8
config.training.n_epochs       = int(args.n_epochs)
config.training.anneal_power   = 2
config.training.log_all_sigmas = False
config.training.eval_freq      = 100 # In steps

# Data
config.data.channels       = 2 # {Re, Im}
config.data.noise_std      = 0.01 # 'Beta' in paper
config.data.image_size     = [args.image_size[0], args.image_size[1]]
config.data.file = args.file

print('Training on Dataset: ' + config.data.file + '\n')

# Get datasets and loaders for channels
dataset     = Knee_Basis_Loader(config)
dataloader  = DataLoader(dataset, batch_size=config.training.batch_size, 
                         shuffle=True, num_workers=config.training.num_workers, 
                         drop_last=True)

# pairwise_dist(config, dataset, tqdm)

config.model.sigma_begin = np.loadtxt(sys.path[0] + '/parameters/' + config.data.file + '.txt')
# config.model.sigma_rate = sigma_rate(dataset, tqdm)
config.model.sigma_rate = args.sigma_rate
config.model.sigma_end  = config.model.sigma_begin * config.model.sigma_rate ** (config.model.num_classes - 1)
config.model.step_size = step_size(config)

print('Sigma Begin: ' + str(config.model.sigma_begin))
print('Sigma Rate: ' + str(config.model.sigma_rate))
print('Sigma End: ' + str(config.model.sigma_end) + '\n')

# Get a model
if args.depth == 'large':
    diffuser = NCSNv2Deepest(config)
elif args.depth == 'medium':
    diffuser = NCSNv2Deeper(config)
elif args.depth == 'low':
    diffuser = NCSNv2(config)

diffuser = diffuser.cuda()
# Get optimizer
optimizer = get_optimizer(config, diffuser.parameters())

# Counter
start_epoch = 0
step = 0
if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(diffuser)

# Get a collection of sigma values
sigmas = get_sigmas(config)

# More logging
config.log_path = 'models/' + config.data.file + '/\
sigma_begin%d_sigma_end%.4f_num_classes%.1f_sigma_rate%.4f_epochs%.1f' % (
    config.model.sigma_begin, config.model.sigma_end,
    config.model.num_classes, config.model.sigma_rate, 
    config.training.n_epochs)

if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)

# No sigma logging
hook = test_hook = None

# Logged metrics
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
        loss, nrmse, nrmse_img = anneal_dsm_score_estimation(
            diffuser, sample['X'], sigmas, None, 
            config.training.anneal_power, hook)
        
        # Keep a running loss
        if step == 1:
            running_loss = loss.item()
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