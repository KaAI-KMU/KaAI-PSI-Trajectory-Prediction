import os
import torch
import json
import cv2
from opts import get_opts
from tqdm import tqdm

from utils import utils

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
args, cfg = get_opts()

# colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255),
#           (0, 102, 153), (255, 255, 0), (255, 255, 255), (153, 153, 255), (255, 0, 255),
#           (128, 0, 255), (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0),
#           (128, 255, 0), (255, 255, 0), (255, 128, 0), (255, 0, 0), (255, 0, 128)]

def validate_intent(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data, training=False)
        intent_prob = torch.sigmoid(intent_logit)
        # intent_pred: logit output, bs
        # traj_pred: logit, bs x ts x 4

        # 1. intent loss
        if args.intent_type == 'mean' and args.intent_num == 2:  # BCEWithLogitsLoss
            gt_intent = data['intention_binary'][:, args.observe_length].type(FloatTensor)
            gt_intent_prob = data['intention_prob'][:, args.observe_length].type(FloatTensor)
            # gt_disagreement = data['disagree_score'][:, args.observe_length]
            # gt_consensus = (1 - gt_disagreement).to(device)

        recorder.eval_intent_batch_update(itern, data, gt_intent.detach().cpu().numpy(),
                                   intent_prob.detach().cpu().numpy(), gt_intent_prob.detach().cpu().numpy())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters}")

    recorder.eval_intent_epoch_calculate(writer)

    return recorder


def test_intent(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    recorder.eval_epoch_reset(epoch, niters)
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data, training=False)
        intent_prob = torch.sigmoid(intent_logit)
        # intent_pred: logit output, bs x 1
        # traj_pred: logit, bs x ts x 4

        # 1. intent loss
        if args.intent_type == 'mean' and args.intent_num == 2:  # BCEWithLogitsLoss
            gt_intent = data['intention_binary'][:, args.observe_length].type(FloatTensor)
            gt_intent_prob = data['intention_prob'][:, args.observe_length].type(FloatTensor)

        recorder.eval_intent_batch_update(itern, data, gt_intent.detach().cpu().numpy(),
                                   intent_prob.detach().cpu().numpy(), gt_intent_prob.detach().cpu().numpy())

    recorder.eval_intent_epoch_calculate(writer)

    return recorder


def predict_intent(model, dataloader, args):
    model.eval()
    dt = {}
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data, training=False)
        intent_prob = torch.sigmoid(intent_logit)

        for i in range(len(data['frames'])):
            vid = data['video_id'][i]  # str list, bs x 60
            pid = data['ped_id'][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent
            # gt_int = data['intention_binary'][i][args.observe_length].item()  # int list, bs x 60
            # gt_int_prob = data['intention_prob'][i][args.observe_length].item()  # float list, bs x 60
            # gt_disgr = data['disagree_score'][i][args.observe_length].item()  # float list, bs x 60
            int_prob = intent_prob[i].item()
            int_pred = round(int_prob) # <0.5 --> 0, >=0.5 --> 1.

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]['intent_pred'] = int_pred
            dt[vid][pid][fid]['intent_pred_prob'] = int_prob

    with open(os.path.join(args.checkpoint_path, 'results', 'test_intent_prediction.json'), 'w') as f:
        json.dump(dt, f)


def validate_traj(model, dataloader, args, recorder, writer):
    total_val_loss = 0
    model.eval()
    niters = len(dataloader)
    for itern, data in enumerate(tqdm(dataloader, desc='Validation')):
        with torch.no_grad():
            result_dict = model(data, training=False)
        traj_pred = result_dict['traj_pred']
        if args.absolute_bbox_input: # traj_gt is relative to the first frame
            traj_gt = data['bboxes'][:,args.observe_length:,:].type(FloatTensor) - data['bboxes'][:,:1,:].type(FloatTensor)
        else:
            traj_gt = data['bboxes'][:,args.observe_length:,:].type(FloatTensor)

        loss_dict = model.get_loss(data['targets'].to(device))
        traj_loss = loss_dict['traj_loss']
        total_val_loss += traj_loss * args.batch_size
        
        min_bbox = torch.tensor(args.min_bbox).type(FloatTensor).to(device)
        max_bbox = torch.tensor(args.max_bbox).type(FloatTensor).to(device)
        traj_pred = utils.convert_unnormalize_bboxes(
            bboxes=traj_pred,
            normalize=args.normalize_bbox,
            # bbox_type='ltrb' if args.bbox_type == 'cxcywh' else None,
            bbox_type2cvt='ltrb' if args.bbox_type == 'cxcywh' else None,
            min_bbox=min_bbox,
            max_bbox=max_bbox,
        )
        traj_gt = utils.convert_unnormalize_bboxes(
            bboxes=traj_gt,
            normalize=args.normalize_bbox,
            # bbox_type='ltrb' if args.bbox_type == 'cxcywh' else None,
            bbox_type2cvt='ltrb' if args.bbox_type == 'cxcywh' else None,
            min_bbox=min_bbox,
            max_bbox=max_bbox,
        )

        recorder.eval_traj_batch_update(itern, data, traj_gt.detach().cpu().numpy(), traj_pred.detach().cpu().numpy())

    val_loss = total_val_loss / len(dataloader)
    score = recorder.eval_traj_epoch_calculate(writer)

    return recorder, score, val_loss


def predict_traj(model, dataloader, args, dset='test'):
    model.eval()
    dt = {}
    for j, data in enumerate(dataloader):
        result_dict = model(data, training=False)
        traj_pred = result_dict['traj_pred']
        if args.absolute_bbox_input: # traj_gt is relative to the first frame
            traj_gt = data['bboxes'][:,args.observe_length:,:].type(FloatTensor) - data['bboxes'][:,:1,:].type(FloatTensor)
        else:
            traj_gt = data['bboxes'][:,args.observe_length:,:].type(FloatTensor)

        min_bbox = torch.tensor(args.min_bbox).type(FloatTensor).to(device)
        max_bbox = torch.tensor(args.max_bbox).type(FloatTensor).to(device)
        traj_pred = utils.convert_unnormalize_bboxes(
            bboxes=traj_pred,
            normalize=args.normalize_bbox,
            bbox_type2cvt='ltrb' if args.bbox_type == 'cxcywh' else None,
            min_bbox=min_bbox,
            max_bbox=max_bbox,
        )
        traj_gt = utils.convert_unnormalize_bboxes(
            bboxes=traj_gt,
            normalize=args.normalize_bbox,
            bbox_type2cvt='ltrb' if args.bbox_type == 'cxcywh' else None,
            min_bbox=min_bbox,
            max_bbox=max_bbox,
        )

        if args.visualize:
            os.makedirs(f"./psi_dataset/frames_dot/{data['video_id'][0]}", exist_ok=True)

            traj_pred_abs = traj_pred + data['original_bboxes'][:,:1,:].type(FloatTensor) if args.absolute_bbox_input else traj_pred
            traj_gt_abs = traj_gt + data['original_bboxes'][:,:1,:].type(FloatTensor) if args.absolute_bbox_input else traj_gt

            traj_pred_vis = utils.convert_bbox(traj_pred_abs, bbox_type='cxcywh')
            traj_gt_vis = utils.convert_bbox(traj_gt_abs, bbox_type='cxcywh')

            for i, (single_traj_pred, single_traj_gt) in enumerate(zip(traj_pred_vis, traj_gt_vis)):
                single_traj_pred = single_traj_pred.detach().cpu().numpy()
                single_traj_gt = single_traj_gt.detach().cpu().numpy()
                video_id = data["video_id"][i]
                frame_id = int(data["frames"][i][14]) + 1
                image = cv2.imread(f"./psi_dataset/frames/{video_id}/{frame_id:03d}.jpg")
                for point_pred, point_gt in zip(single_traj_pred, single_traj_gt):
                    cv2.circle(image, (int(point_pred[0]), int(point_pred[1])), 1, (0,255,0), -1)
                    cv2.circle(image, (int(point_gt[0]), int(point_gt[1])), 1, (0,0,255), -1)
                # draw gt bbox
                single_traj_gt_ltrb = utils.convert_bbox(single_traj_gt, bbox_type='ltrb')
                first_bbox = single_traj_gt_ltrb[0]
                cv2.rectangle(image, (int(first_bbox[0]), int(first_bbox[1])), (int(first_bbox[2]), int(first_bbox[3])), (255,0,0), 1)
                cv2.imwrite(f"./psi_dataset/frames_dot/{video_id}/{frame_id:03d}.jpg", image)

        for i in range(len(data['frames'])): # for each sample in a batch
            vid = data['video_id'][i]  # str list, bs x 60
            pid = data['ped_id'][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]['traj'] = traj_pred[i].detach().cpu().numpy().tolist()
            # print(len(traj_pred[i].detach().cpu().numpy().tolist()))
    # print("saving prediction...")
    with open(os.path.join(args.checkpoint_path, 'results', f'{dset}_traj_pred.json'), 'w') as f:
        json.dump(dt, f)



def get_test_traj_gt(model, dataloader, args, dset='test'):
    model.eval()
    gt = {}
    for itern, data in enumerate(dataloader):
        output = model(data, training=False)
        traj_pred = output['traj_pred']
        if args.absolute_bbox_input:
            traj_gt = data['bboxes'][:,args.observe_length:,:].type(FloatTensor) - data['bboxes'][:,:1,:].type(FloatTensor)
        else:
            traj_gt = data['bboxes'][:,args.observe_length:,:].type(FloatTensor)
        
        min_bbox = torch.tensor(args.min_bbox).type(FloatTensor).to(device)
        max_bbox = torch.tensor(args.max_bbox).type(FloatTensor).to(device)
        traj_pred = utils.convert_unnormalize_bboxes(
            bboxes=traj_pred,
            normalize=args.normalize_bbox,
            # bbox_type='ltrb' if args.bbox_type == 'cxcywh' else None,
            bbox_type2cvt='ltrb' if args.bbox_type == 'cxcywh' else None,
            min_bbox=min_bbox,
            max_bbox=max_bbox,
        )
        traj_gt = utils.convert_unnormalize_bboxes(
            bboxes=traj_gt,
            normalize=args.normalize_bbox,
            # bbox_type='ltrb' if args.bbox_type == 'cxcywh' else None,
            bbox_type2cvt='ltrb' if args.bbox_type == 'cxcywh' else None,
            min_bbox=min_bbox,
            max_bbox=max_bbox,
        )

        for i in range(len(data['frames'])): # for each sample in a batch
            vid = data['video_id'][i]  # str list, bs x 60
            pid = data['ped_id'][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if vid not in gt:
                gt[vid] = {}
            if pid not in gt[vid]:
                gt[vid][pid] = {}
            if fid not in gt[vid][pid]:
                gt[vid][pid][fid] = {}
            gt[vid][pid][fid]['traj'] = traj_gt[i].detach().cpu().numpy().tolist()
            # print(len(traj_pred[i].detach().cpu().numpy().tolist()))
    with open(os.path.join(f'./test_gt/{dset}_traj_gt.json'), 'w') as f:
        json.dump(gt, f)