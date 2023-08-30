import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_template import ModelTemplate

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class GRUTrajBbox(ModelTemplate):
    def __init__(self, model_cfg, dataset):
        super(GRUTrajBbox, self).__init__()
        model_opts = None
        self.enc_in_dim = model_cfg['enc_in_dim']  # 4, input bbox+convlstm_output context vector
        self.enc_out_dim = model_cfg['enc_out_dim']  # 64
        self.dec_in_emb_dim = model_cfg['dec_in_emb_dim']  # 1 for intent, 1 for speed, ? for rsn
        self.dec_out_dim = model_cfg['dec_out_dim']  # 64 for lstm decoder output
        self.output_dim = model_cfg['output_dim']
        
        n_layers = model_opts['n_layers']
        dropout = model_opts['dropout']
        predict_length = model_opts['predict_length']
        self.predict_length = predict_length
                
        self.encoder = nn.GRU(
            input_size=self.enc_in_dim,
            hidden_size=self.enc_out_dim,
        )

        self.dec_in_dim = self.enc_out_dim + self.dec_in_emb_dim
        self.decoder = nn.GRU(
            input_size=self.dec_in_dim,
            hidden_size=self.dec_out_dim,
            num_layers=n_layers,
            batch_first=True,
            bias=True
        )

        self.fc = nn.Sequential(
            nn.Linear(self.dec_out_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, self.output_dim)
        )
        
    def forward(self, data, training=True):
        bbox = data['bboxes'][:, :self.observe_length, :].type(FloatTensor)
        enc_input = bbox

        enc_output, enc_h = self.encoder(enc_input)
        enc_last_output = enc_output[:, -1:, :]

        traj_pred_list = []
        prev_hidden = enc_h

        dec_input_emb = None

        for t in range(self.predict_length):
            if dec_input_emb is None:
                dec_input = enc_last_output
            else:
                dec_input = torch.cat([enc_last_output, dec_input_emb[:, t, :].unsqueeze(1)])

            dec_output, prev_hidden = self.decoder(dec_input, prev_hidden)
            logit = self.fc(dec_output.squeeze(1))
            traj_pred_list.append(logit)

        traj_pred = torch.stack(traj_pred_list, dim=0).transpose(1, 0)

        return traj_pred
    
    def get_loss(self, targets):
        targets = targets.type(FloatTensor)
        traj_loss = self.criterion(self.forward_ret_dict['traj_pred'], targets)
        loss_dict = {
            'traj_loss': traj_loss
            }
        return loss_dict

    def build_optimizer(self, optim_cfg, scheduler_cfg):
        optimizer = torch.optim.Adam(self.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay, eps=optim_cfg.eps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_cfg.gamma)

        return optimizer, scheduler