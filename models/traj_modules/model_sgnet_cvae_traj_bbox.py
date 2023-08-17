import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import rmse_loss, cvae_multi, euclidean_dist
from ..model_template import ModelTemplate
from .model_bitrap_traj_bbox import BiTraPNP
from ..feature_extractor import *

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class SGNetCVAETrajBbox(ModelTemplate):
    def __init__(self, model_cfg, dataset):
        super(SGNetCVAETrajBbox, self).__init__()
        self.cvae = BiTraPNP(model_cfg.cvae)
        self.hidden_size = model_cfg.hidden_size # GRU hidden size
        self.enc_steps = model_cfg.enc_steps # observation step
        self.dec_steps = model_cfg.dec_steps # prediction step
        self.dataset = dataset
        self.dropout = model_cfg.dropout
        self.use_speed = model_cfg.get('speed_module', None) is not None
        self.use_desc = model_cfg.get('description_module', None) is not None
        
        self.bbox_module = SgnetFeatureExtractor(model_cfg.bbox_module)
        self.speed_module = SgnetFeatureExtractor(model_cfg.speed_module) if self.use_speed else None
        self.desc_module = DescFeatureExtractor(model_cfg.description_module, freeze_params=True) if self.use_desc else None
        
        self.pred_dim = model_cfg.pred_dim
        self.K = model_cfg.K
        self.map = False
        if self.dataset in ['PSI2.0','JAAD','PIE']:
            # the predict shift is in pixel
            self.pred_dim = 4
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                     self.pred_dim),
                                                     nn.Tanh())
            self.ed_estimator = nn.Sequential(nn.Linear(self.hidden_size,
                                                        1))
            self.flow_enc_cell = nn.GRUCell(self.hidden_size*2, self.hidden_size)
        elif self.dataset in ['ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            self.pred_dim = 2
            # the predict shift is in meter
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                        self.pred_dim))   
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))

        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.cvae_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size + model_cfg.latent_dim,
                                                self.hidden_size),
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
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)
        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)

        self.criterion = rmse_loss().to(device)
        self.criterion_ed = nn.CrossEntropyLoss().to(device)
        self.observe_length = model_cfg.observe_length
        self.forward_ret_dict = {}
    
    def SGE(self, goal_hidden):
        # initial goal input with zero
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
        # initial trajectory tensor
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            # next step input is generate by hidden
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            # regress goal traj for loss
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list],dim = 1)
        enc_attn= self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim =1).unsqueeze(1)
        goal_for_enc  = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def cvae_decoder(self, dec_hidden, goal_for_dec):
        batch_size = dec_hidden.size(0)
       
        K = dec_hidden.shape[1]
        dec_hidden = dec_hidden.view(-1, dec_hidden.shape[-1])
        dec_traj = dec_hidden.new_zeros(batch_size, self.dec_steps, K, self.pred_dim)
        dec_ed_est = dec_hidden.new_zeros(batch_size, self.dec_steps, K)
        for dec_step in range(self.dec_steps):
            # incremental goal for each time step
            goal_dec_input = dec_hidden.new_zeros(batch_size, self.dec_steps, self.hidden_size//4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:],dim=1)
            goal_dec_input[:,dec_step:,:] = goal_dec_input_temp
            dec_attn= self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim =1).unsqueeze(1)
            goal_dec_input  = torch.bmm(dec_attn,goal_dec_input).squeeze(1)
            goal_dec_input = goal_dec_input.unsqueeze(1).repeat(1, K, 1).view(-1, goal_dec_input.shape[-1])
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input,dec_dec_input),dim = -1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            batch_traj = self.regressor(dec_hidden)
            batch_traj = batch_traj.view(-1, K, batch_traj.shape[-1])
            dec_traj[:,dec_step,:,:] = batch_traj
            # regress dec conf for loss
            batch_ed_est = self.ed_estimator(dec_hidden).view(-1, K)
            dec_ed_est[:,dec_step,:] = batch_ed_est
        return dec_traj, dec_ed_est

    def encoder(self, raw_inputs, raw_targets, traj_input, flow_input=None, start_index = 0):
        # initial output tensor
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_cvae_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.K, self.pred_dim)
        total_ed_est = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.K)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        total_probabilities = traj_input.new_zeros((traj_input.size(0), self.enc_steps, self.K))
        total_KLD = 0
        for enc_step in range(start_index, self.enc_steps):
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            enc_hidden = traj_enc_hidden
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            all_goal_traj[:,enc_step,:,:] = goal_traj
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)
            if self.training:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K, raw_targets[:,enc_step,:,:])
            else:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K)
            total_probabilities[:,enc_step,:] = probability
            total_KLD += KLD
            cvae_dec_hidden= self.cvae_to_dec_hidden(cvae_hidden)
            if self.map:
                map_input = flow_input
                cvae_dec_hidden = (cvae_dec_hidden + map_input.unsqueeze(1))/2
            cvae_dec_traj_single, ed_est_single = self.cvae_decoder(cvae_dec_hidden, goal_for_dec)
            all_cvae_dec_traj[:,enc_step,:,:,:] = cvae_dec_traj_single
            total_ed_est[:,enc_step,:,:] = ed_est_single
        return all_goal_traj, all_cvae_dec_traj, total_KLD, total_ed_est
            
    def forward(self, data, training=True):
        bboxes = data['bboxes'][:,:self.observe_length,:].to(device).type(FloatTensor)
        targets = data['targets'].to(device).type(FloatTensor)
        self.training = training
        if torch.is_tensor(0):
            start_index = start_index[0].item()
        if self.dataset in ['PSI2.0','JAAD','PIE']:
            input_list = []
            bbox_input = self.bbox_module(bboxes)
            input_list.append(bbox_input)
            if self.use_speed:
                speed = data['speed'][:, :self.observe_length, :].to(device).type(FloatTensor)
                speed_input = self.speed_module(speed)
                input_list.append(speed_input)
            if self.use_desc:
                desc = data['single_description']
                desc_input = self.desc_module(desc, frame_len=bbox_input.shape[1])
                input_list.append(desc_input)
            traj_input = torch.cat(input_list, dim=-1)
            all_goal_traj, all_cvae_dec_traj, KLD, total_ed_est = self.encoder(bboxes, targets, traj_input)
        elif self.dataset in ['ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            traj_input_temp = self.bbox_module(bboxes[:,start_index:,:])
            traj_input = traj_input_temp.new_zeros((bboxes.size(0), bboxes.size(1), traj_input_temp.size(-1)))
            traj_input[:,start_index:,:] = traj_input_temp
            all_goal_traj, all_cvae_dec_traj, KLD, total_ed_est = self.encoder(bboxes, targets, traj_input, None, start_index)
        
        self.forward_ret_dict['all_goal_traj'] = all_goal_traj
        self.forward_ret_dict['all_cvae_dec_traj'] = all_cvae_dec_traj # (bs, observe_length, predict_length, K, 4)
        self.forward_ret_dict['KLD'] = KLD
        self.forward_ret_dict['total_ed_est'] = total_ed_est

        best_indices = torch.argmax(-total_ed_est, dim=-1).to(device)[:,-1,:] # (bs, K)
        bs, pl = best_indices.shape
        traj_pred = all_cvae_dec_traj[torch.arange(bs)[:, None], -1, torch.arange(pl)[None, :], best_indices, :].squeeze()
        self.forward_ret_dict['traj_pred'] = traj_pred

        return self.forward_ret_dict
    
    def get_loss(self, targets):
        all_goal_traj = self.forward_ret_dict['all_goal_traj']
        all_cvae_dec_traj = self.forward_ret_dict['all_cvae_dec_traj']
        KLD = self.forward_ret_dict['KLD']
        total_ed_est = self.forward_ret_dict['total_ed_est']

        cvae_pred = all_cvae_dec_traj
        traj_pred = all_goal_traj
        kld_loss = KLD.mean()
        cvae_loss = cvae_multi(cvae_pred, targets)
        goal_loss = self.criterion(traj_pred, targets)
        total_ed = euclidean_dist(all_cvae_dec_traj, targets.unsqueeze(3).repeat(1,1,1,self.K,1))
        ed_loss = self.criterion_ed(total_ed_est, total_ed)

        traj_loss = kld_loss + cvae_loss + goal_loss + ed_loss

        loss_dict = {
            'traj_loss': traj_loss,
            'kld_loss': kld_loss,
            'cvae_loss': cvae_loss,
            'goal_loss': goal_loss,
            'ed_loss': ed_loss,
        }

        return loss_dict
        