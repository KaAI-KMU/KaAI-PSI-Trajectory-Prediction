import collections

from test import validate_traj
from tqdm import tqdm
import torch
import numpy as np
import os


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def train_traj(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer):
    pos_weight = torch.tensor(args.intent_positive_weight).to(device)
    epoch_loss = {'loss_intent': [], 'loss_traj': []}

    best_val_score = float('inf') #최선
    
    for epoch in tqdm(range(1, args.epochs+1), desc="Training Epoch"):
        niters = len(train_loader)
        recorder.train_epoch_reset(epoch, niters)
        epoch_loss = train_traj_epoch(epoch, model, optimizer, epoch_loss, train_loader, args, recorder, writer)

        if epoch % args.val_freq == 0:
            niters = len(val_loader)
            recorder.eval_epoch_reset(epoch, niters)
            _, val_score, val_loss = validate_traj(model, val_loader, args, recorder, writer) #backbone_model) 
            if val_score < best_val_score:
                print(f"New best validation score ({val_score}), saving model...")
                best_val_score = val_score
                torch.save(model.state_dict(), args.checkpoint_path + f'/best.pth')

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
            scheduler.step()

        torch.save(model.state_dict(), args.checkpoint_path + f'/latest.pth')
       


def train_traj_epoch(epoch, model, optimizer, epoch_loss, dataloader, args, recorder, writer):
    model.train()
    batch_losses = collections.defaultdict(list)

    niters = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=niters, desc=f"train")
    for itern, data in pbar:
        result_dict = model(data, training=True)
        loss_dict = model.get_loss(data['targets'].to(device))
        traj_loss = loss_dict['traj_loss']
        traj_pred = result_dict['traj_pred']
        traj_gt = data['bboxes'][:, args.observe_length:, :].type(FloatTensor)
        
        loss = args.loss_weights['loss_traj'] * traj_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record results
        batch_losses['loss'].append(loss.item())
        batch_losses['loss_traj'].append(traj_loss.item())

        pbar.set_postfix({"loss_traj": f"{np.mean(batch_losses['loss_traj']): .4f}"})

        recorder.train_traj_batch_update(itern, data, traj_gt.detach().cpu().numpy(), traj_pred.detach().cpu().numpy(),
                                         loss.item(), traj_loss.item())
        break
        

    epoch_loss['loss_traj'].append(np.mean(batch_losses['loss_traj']))

    recorder.train_traj_epoch_calculate(writer)
    # write scalar to tensorboard
    writer.add_scalar(f'LearningRate', optimizer.param_groups[-1]['lr'], epoch)
    for key, val in batch_losses.items():
        writer.add_scalar(f'Losses/{key}', np.mean(val), epoch)

    return epoch_loss