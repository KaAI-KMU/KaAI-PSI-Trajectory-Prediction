import os
import math

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

class ModelTemplate(nn.Module):
    def __init__(self):
        super(ModelTemplate, self).__init__()
    
    def forward(self, data, training=False):
        raise NotImplementedError
    
    def get_loss(self, targets):
        raise NotImplementedError
        
    def build_optimizer(self, optim_cfg, scheduler_cfg):
        optimizer = torch.optim.Adam(self.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
        if scheduler_cfg.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_cfg.factor,
                                                                   patience=scheduler_cfg.patience, min_lr=scheduler_cfg.min_lr,
                                                                   verbose=True)
        elif scheduler_cfg.scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_cfg.gamma)
        elif scheduler_cfg.scheduler == 'CosineAnealingWarmUpRestarts':
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=scheduler_cfg.T_0, T_mult=scheduler_cfg.T_mult,
                                                      eta_max=scheduler_cfg.eta_max, T_up=scheduler_cfg.T_up,
                                                      gamma=scheduler_cfg.gamma)
        elif scheduler_cfg.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma)
        elif scheduler_cfg.scheduler == 'none':
            scheduler = None
        else:
            raise NotImplementedError
        return optimizer, scheduler
        
    def load_params_from_file(self, filename, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint
        
        # Modify this section to include any pre-trained weights if needed
        pretrain_checkpoint = torch.load(filename, map_location=loc_type)
        pretrain_model_state_disk = pretrain_checkpoint
        model_state_disk.update(pretrain_model_state_disk)
        
        version = checkpoint.get("version", None)
        if version is not None:
            print('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                print('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        print('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))
        
    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        self.layers = []
        for attr_name, attr_value in self.__dict__.items():
            if attr_name == '_modules':
                for module_name in attr_value:
                    self.layers.append(attr_value[module_name])

        # Update the state_dict based on the model_state_disk
        update_model_state = {}
        for key, val in model_state_disk.items():
            if key == 'flow_fc.0.weight':
                key='flow_fc.weight'
            elif key == 'flow_fc.0.bias':
                key='flow_fc.bias'

            if key.startswith('module.'):
                key = key[key.find('.')+1:]
            if 'feature_extractor' in key: # feature_extractor is renamed to bbox_module
                key = key.replace('feature_extractor', 'bbox_module')
            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                print(f"Unrecognized key: {key}") 

        # Update the model's state_dict
        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)

        return state_dict, update_model_state
    
    
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr