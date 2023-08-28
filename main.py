from opts import get_opts
from datetime import datetime
import os
import json
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.prepare_data import get_dataloader
from database.create_database import create_database
from models.build_model import build_model
from train import train_traj
from test import predict_traj, get_test_traj_gt
from utils.log import RecordResults
from utils.evaluate_results import evaluate_traj


def main(args, config):
    writer = SummaryWriter(args.checkpoint_path)
    recorder = RecordResults(args)
    ''' 1. Load database '''
    if not os.path.exists(os.path.join(args.database_path, args.database_file)):
        create_database(args)
    else:
        print("Database exists!")
    train_loader, val_loader, test_loader = get_dataloader(args)

    ''' 2. Create models '''
    model, optimizer, scheduler = build_model(config, pretrained_path=args.pretrained_path)

    if args.pretrained_path is not None:
        val_gt_file = './test_gt/val_traj_gt.json'
        if not os.path.exists(val_gt_file):
            get_test_traj_gt(model, val_loader, args, dset='val')
        predict_traj(model, val_loader, args, dset='val')
        score = evaluate_traj(val_gt_file, args.checkpoint_path + '/results/val_traj_pred.json', args)

    else:
        # ''' 3. Train '''
        train_traj(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer)

        val_gt_file = './test_gt/val_traj_gt.json'
        if not os.path.exists(val_gt_file):
            get_test_traj_gt(model, val_loader, args, dset='val')
        predict_traj(model, val_loader, args, dset='val')
        score = evaluate_traj(val_gt_file, args.checkpoint_path + '/results/val_traj_prediction.json', args)

        # ''' 4. Test '''
        # test_gt_file = './test_gt/test_traj_gt.json'
        # if not os.path.exists('./test_gt/test_traj_gt.json'):
        #     get_test_traj_gt(model, test_loader, args, dset='test')
        # predict_traj(model, test_loader, args, dset='test')
        # score = evaluate_traj(test_gt_file, args.checkpoint_path + '/results/test_traj_prediction.json', args)
        # print("Ranking score of test set: ", score)

if __name__ == '__main__':
    args, cfg = get_opts()
    # Dataset
    if cfg.dataset == 'PSI2.0':
        args.video_splits = os.path.join(args.dataset_root_path, 'PSI2.0_TrainVal/splits/PSI2_split.json')
    elif args.dataset == 'PSI1.0':
        args.video_splits = os.path.join(args.dataset_root_path, 'PSI1.0/splits/PSI1_split.json')
    else:
        raise Exception("Unknown dataset name!")

    # intent prediction
    if args.task_name == 'ped_intent':
        args.database_file = 'intent_database_train.pkl' if args.database_file is None else args.database_file
        args.intent_model = True
        args.traj_model = False
    # trajectory prediction
    elif args.task_name == 'ped_traj':
        args.database_file = 'traj_database_train.pkl' if args.database_file is None else args.database_file
        args.intent_model = False # if use intent prediction module to support trajectory prediction
        args.traj_model = True
    else:
        args.intent_model = False
        args.traj_model = False
        raise Exception("Unknown task name!")

    args.max_bbox = [args.image_shape[0], args.image_shape[1], args.image_shape[0], args.image_shape[1]]
    args.min_bbox = [0, 0, 0, 0]
    # Record
    now = datetime.now()
    time_folder = now.strftime('%Y%m%d%H%M%S')
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.task_name, cfg.dataset, cfg.model_name, time_folder)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    with open(os.path.join(args.checkpoint_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    result_path = os.path.join(args.checkpoint_path, 'results')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    main(args, cfg)