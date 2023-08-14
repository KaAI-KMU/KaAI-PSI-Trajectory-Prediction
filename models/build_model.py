import numpy as np
import torch
import os
from argparse import Namespace
from .traj_modules.model_sgnet_traj_bbox import SGNetTrajBbox
from .traj_modules.model_lstm_traj_bbox import LSTMTrajBbox

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

__all__ = {
    'SGNetTrajBbox': SGNetTrajBbox,
    'LSTMTrajBbox': LSTMTrajBbox
}

def build_model(config):
    model = __all__[config.model_name](model_cfg=config.model_cfg, dataset=config.dataset).to(device)
    optimizer, scheduler = model.build_optimizer(config.optimization, config.lr_scheduler)
    return model, optimizer, scheduler