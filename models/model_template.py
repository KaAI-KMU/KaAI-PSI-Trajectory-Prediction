import os
import torch
from torch import nn

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
        elif scheduler_cfg.scheduler == 'none':
            scheduler = None
        else:
            raise NotImplementedError
        return optimizer, scheduler
        
    def load_params_from_file(self, filename, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint
        
        # Modify this section to include any pre-trained weights if needed
        if pre_trained_path is not None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
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
            # key = key[key.find('.')+1:]
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