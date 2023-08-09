import numpy as np
import torch
import os
from argparse import Namespace
from .traj_modules.model_sgnet_traj_bbox import SGNetTrajBbox

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def build_model(args):
    if args.model_name == 'sgnet_traj_bbox':
        model = get_sgnet_traj_bbox(args).to(device)
        optimizer, scheduler = model.build_optimizer(args)
        return model, optimizer, scheduler
    else:
        raise NotImplementedError

def get_sgnet_traj_bbox(args):
    model_configs = {}
    model_configs['traj_model_opts'] = Namespace(**{
        'hidden_size': 512,
        'enc_steps': 15,
        'dec_steps': 45,
        'dropout': 0.0,
        'nu': 0.0,
        'sigma': 1.5,
        'pred_dim': 4,
        'input_dim': 4,
        'LATENT_DIM': 32,
        'DEC_WITH_Z': True,
        'dataset': args.dataset,
        'K': 20,
        'observe_length': args.observe_length,
    })
    model = SGNetTrajBbox(model_configs['traj_model_opts'])
    return model