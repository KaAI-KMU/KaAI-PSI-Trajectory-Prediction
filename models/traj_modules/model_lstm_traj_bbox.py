import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_template import ModelTemplate

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class LSTMTrajBbox(ModelTemplate):
    def __init__(self, model_cfg, dataset):
        super(LSTMTrajBbox, self).__init__()

        self.enc_in_dim = model_cfg.enc_in_dim  # 4, input bbox+convlstm_output context vector
        self.enc_out_dim = model_cfg.enc_out_dim  # 64
        self.dec_in_emb_dim = model_cfg.dec_in_emb_dim  # 1 for intent, 1 for speed, ? for rsn
        self.dec_out_dim = model_cfg.dec_out_dim  # 64 for lstm decoder output
        self.output_dim = model_cfg.output_dim  # 4 for bbox, 2/3: intention; 62 for reason; 1 for trust score; 4 for trajectory.

        n_layers = model_cfg.n_layers
        dropout = model_cfg.dropout
        self.observe_length = model_cfg.observe_length
        self.predict_length = model_cfg.predict_length
        self.dataset = dataset

        self.backbone = None

        self.encoder = nn.LSTM(
            input_size=self.enc_in_dim,
            hidden_size=self.enc_out_dim,
            num_layers=n_layers,
            batch_first=True,
            bias=True
        )

        self.dec_in_dim = self.enc_out_dim + self.dec_in_emb_dim
        self.decoder = nn.LSTM(
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

        if model_cfg['output_activation'] == 'tanh':
            self.activation = nn.Tanh()
        elif model_cfg['output_activation'] == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

        self.module_list = [self.encoder, self.decoder, self.fc] #, self.fc_emb, self.decoder
        self.forward_ret_dict = {}
        self.criterion = nn.MSELoss()

    def forward(self, data, training=True):
        bbox = data['bboxes'][:, :self.observe_length, :].type(FloatTensor)
        # enc_input/dec_input_emb: bs x ts x enc_input_dim/dec_emb_input_dim
        enc_input = bbox

        # 1. encoder
        enc_output, (enc_hc, enc_nc) = self.encoder(enc_input)
        # because 'batch_first=True'
        # enc_output: bs x ts x (1*hiden_dim)*enc_hidden_dim --- only take the last output, concatenated with dec_input_emb, as input to decoder
        # enc_hc:  (n_layer*n_directions) x bs x enc_hidden_dim
        # enc_nc:  (n_layer*n_directions) x bs x enc_hidden_dim
        enc_last_output = enc_output[:, -1:, :]  # bs x 1 x hidden_dim

        # 2. decoder
        traj_pred_list = []
        evidence_list = []
        prev_hidden = enc_hc
        prev_cell = enc_nc

        dec_input_emb = None
        # if self.intent_embedding:
        #     # shape: (bs,)
        #     intent_gt_prob = data['intention_prob'][:, self.args.observe_length].type(FloatTensor)
        #     intent_pred = data['intention_pred'].type(FloatTensor) # bs x 1

        for t in range(self.predict_length):
            if dec_input_emb is None:
                dec_input = enc_last_output
            else:
                dec_input = torch.cat([enc_last_output, dec_input_emb[:, t, :].unsqueeze(1)])

            dec_output, (dec_hc, dec_nc) = self.decoder(dec_input, (prev_hidden, prev_cell))
            logit = self.fc(dec_output.squeeze(1)) # bs x 4
            traj_pred_list.append(logit)
            prev_hidden = dec_hc
            prev_cell = dec_nc

        traj_pred = torch.stack(traj_pred_list, dim=0).transpose(1, 0) # ts x bs x 4 --> bs x ts x 4
        self.forward_ret_dict = {'traj_pred': traj_pred}

        return self.forward_ret_dict
    
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

    def lr_scheduler(self, optimizer, cur_epoch, args, gamma=10, power=0.75):
        decay = (1 + gamma * cur_epoch / args.epochs) ** (-power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)