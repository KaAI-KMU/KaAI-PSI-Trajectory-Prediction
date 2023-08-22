import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import rmse_loss
from ..model_template import ModelTemplate
from ..feature_extractor import *

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class SGNetTrajBbox(ModelTemplate):
    def __init__(self, model_cfg, dataset=None):
        super(SGNetTrajBbox, self).__init__()

        self.observe_length = model_cfg.observe_length
        self.hidden_size = model_cfg.hidden_size
        self.enc_steps = model_cfg.enc_steps
        self.dec_steps = model_cfg.dec_steps
        self.dropout = model_cfg.dropout
        self.use_speed = model_cfg.get('speed_module', None) is not None
        self.use_flow = model_cfg.get('flow_module', None) is not None

        self.bbox_module = SgnetFeatureExtractor(model_cfg.bbox_module)
        traj_enc_cell_hidden_size = model_cfg.bbox_module.output_dim + self.hidden_size//4
        dec_cell_hidden_size = self.hidden_size + self.hidden_size//4
        if self.use_speed:
            self.speed_module = SgnetFeatureExtractor(model_cfg.speed_module)
            self.speed_fc = nn.Linear(self.observe_length, 1)
            dec_cell_hidden_size += model_cfg.speed_module.output_dim
        else:
            self.speed_module = None
            self.speed_fc = None
        if self.use_flow:
            self.flow_module = SgnetFeatureExtractor(model_cfg.flow_module) if self.use_flow else None
            self.flow_fc = nn.Linear(self.observe_length, 1)
            dec_cell_hidden_size += model_cfg.flow_module.output_dim
        else:
            self.flow_module = None
            self.flow_fc = None

        self.pred_dim = 4
        self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                    self.pred_dim),
                                                    nn.Tanh())
        self.flow_enc_cell = nn.GRUCell(self.hidden_size*2, self.hidden_size)
             
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))

        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size),
                                                nn.ReLU(inplace=True))


        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)

        self.traj_enc_cell = nn.GRUCell(traj_enc_cell_hidden_size, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        self.dec_cell = nn.GRUCell(dec_cell_hidden_size, self.hidden_size)

        self.criterion = rmse_loss().to(device)
        self.forward_ret_dict = {}        

    def SGE(self, goal_hidden):
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            # regress goal traj for loss
            goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list],dim = 1)
        enc_attn= self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim =1).unsqueeze(1)
        goal_for_enc  = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def decoder(self, dec_hidden, goal_for_dec, additional_dict=None):
        # initial trajectory tensor
        dec_traj = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.pred_dim)
        for dec_step in range(self.dec_steps):
            goal_dec_input = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.hidden_size//4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:],dim=1)
            goal_dec_input[:,dec_step:,:] = goal_dec_input_temp
            dec_attn= self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim =1).unsqueeze(1)
            goal_dec_input  = torch.bmm(dec_attn,goal_dec_input).squeeze(1)#.view(goal_hidden.size(0), self.dec_steps, self.hidden_size//4).sum(1)
            
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input,dec_dec_input),dim = -1))
            if additional_dict is not None:
                for k, input_features in additional_dict.items():
                    dec_input = torch.cat((dec_input, input_features), dim=-1)
            # regress dec traj for loss
            dec_traj[:,dec_step,:] = self.regressor(dec_hidden)
        return dec_traj
        
    def encoder(self, traj_input, additional_dict=None):
        # initial output tensor
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        for enc_step in range(0, self.enc_steps):
            
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            enc_hidden = traj_enc_hidden
            # generate hidden states for goal and decoder 
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)

            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            if additional_dict is not None:
                additional_dict_single = {k: v[:,enc_step,:] for k, v in additional_dict.items()}
            dec_traj = self.decoder(dec_hidden, goal_for_dec, additional_dict_single)

            # output 
            all_goal_traj[:,enc_step,:,:] = goal_traj
            all_dec_traj[:,enc_step,:,:] = dec_traj
        
        return all_goal_traj, all_dec_traj
            

    def forward(self, data, training=True):
        bboxes = data['bboxes'][:,:self.observe_length,:].to(device).type(FloatTensor)

        traj_input = self.bbox_module(bboxes)
        additional_dict = {}
        if self.use_speed:
            speed = data['speed'][:, :self.observe_length, :].to(device).type(FloatTensor)
            speed_input = self.speed_module(speed)
            additional_dict['speed_input'] = speed_input
        if self.use_flow:
            flow = data['optical_flow'][:, :self.observe_length, :].to(device).type(FloatTensor)
            flow_input = self.flow_module(flow)
            additional_dict['flow_input'] = flow_input

        all_goal_traj, all_dec_traj = self.encoder(traj_input, additional_dict)

        self.forward_ret_dict['all_goal_traj'] = all_goal_traj
        self.forward_ret_dict['all_dec_traj'] = all_dec_traj
        self.forward_ret_dict['traj_pred'] = all_dec_traj[:,-1,:,:]
        return self.forward_ret_dict
    
    def get_loss(self, targets):
        goal_loss = self.criterion(self.forward_ret_dict['all_goal_traj'], targets)
        dec_loss = self.criterion(self.forward_ret_dict['all_dec_traj'], targets)
        traj_loss = goal_loss + dec_loss
        return {'traj_loss': traj_loss, 'goal_loss': goal_loss, 'dec_loss': dec_loss}