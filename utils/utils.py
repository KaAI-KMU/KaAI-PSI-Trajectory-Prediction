import numpy as np
import torch
import torch.nn as nn

def save_args(self, args):
    # 3. args
    with open(args.checkpoint_path + '/args.txt', 'w') as f:
        for arg in vars(self.args):
            val = getattr(self.args, arg)
            if isinstance(val, str):
                val = f"'{val}'"
            f.write("{}: {}\n".format(arg, val))
    np.save(self.args.checkpoint_path + "/args.npy", self.args)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class rmse_loss(nn.Module):
    '''
    Params:
        x_pred: (batch_size, enc_steps, dec_steps, pred_dim)
        x_true: (batch_size, enc_steps, dec_steps, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''
    def __init__(self):
        super(rmse_loss, self).__init__()
    
    def forward(self, x_pred, x_true):
        L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=3))
        # sum over prediction time steps
        L2_all_pred = torch.sum(L2_diff, dim=2)
        # mean of each frames predictions
        L2_mean_pred = torch.mean(L2_all_pred, dim=1)
        # sum of all batches
        L2_mean_pred = torch.mean(L2_mean_pred, dim=0)
        return L2_mean_pred
    

def cvae_multi(pred_traj, target, first_history_index = 0):
    K = pred_traj.shape[3]
    
    target = target.unsqueeze(3).repeat(1, 1, 1, K, 1)
    total_loss = []
    for enc_step in range(first_history_index, pred_traj.size(1)):
        traj_rmse = torch.sqrt(torch.sum((pred_traj[:,enc_step,:,:,:] - target[:,enc_step,:,:,:])**2, dim=-1)).sum(dim=1)
        best_idx = torch.argmin(traj_rmse, dim=1)
        loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
        total_loss.append(loss_traj)
    
    return sum(total_loss)/len(total_loss)