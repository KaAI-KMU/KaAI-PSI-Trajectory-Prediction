from opts import get_opts
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from data.prepare_data import get_dataloader
from database.create_database import create_database


def generate_center_of_single(output_path, video_id, frame_id, ped_id, optical_flow):
    # save center of optical flow as numpy
    for i in range(optical_flow.shape[0]):
        save_dir = output_path + f'/{video_id}/{frame_id[i]}'
        save_path = save_dir + f'_{ped_id}.npy'
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        np.save(save_path, optical_flow)

def generate_center_of(output_path, data_loader, dset='train'):
    for data in tqdm(data_loader, desc=f"Generating center optical flow of {dset} dataset"):
        video_ids = data['video_id']
        frame_ids = data['frames']
        ped_ids = data['ped_id']
        optical_flows = data['optical_flow']

        # save center of optical flow as numpy
        batch_size = len(video_ids)
        for bs in range(batch_size):
            video_id = video_ids[bs]
            frame_id = frame_ids[bs]
            ped_id = ped_ids[bs]
            optical_flow = optical_flows[bs].numpy()
            generate_center_of_single(output_path, video_id, frame_id, ped_id, optical_flow)


def main(args, config):
    ''' 1. Load database '''
    if not os.path.exists(os.path.join(args.database_path, args.database_file)):
        create_database(args)
    else:
        print("Database exists!")
    args.gen_center_of = True
    train_loader, val_loader, test_loader = get_dataloader(args)
    output_path = os.path.join(args.dataset_root_path, 'center_of')

    generate_center_of(output_path, train_loader, dset='train')
    generate_center_of(output_path, val_loader, dset='val')
    # generate_center_of(output_path, test_loader, args, dset='test')


if __name__ == '__main__':
    args, cfg = get_opts()
    # Dataset
    args.video_splits = os.path.join(args.dataset_root_path, 'PSI2.0_TrainVal/splits/PSI2_split.json')

    args.database_file = 'traj_database_train.pkl' if args.database_file is None else args.database_file
    args.intent_model = False # if use intent prediction module to support trajectory prediction
    args.traj_model = True

    args.max_bbox = [args.image_shape[0], args.image_shape[1], args.image_shape[0], args.image_shape[1]]
    args.min_bbox = [0, 0, 0, 0]

    main(args, cfg)