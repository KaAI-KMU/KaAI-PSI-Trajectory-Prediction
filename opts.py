import argparse
import yaml

from easydict import EasyDict

# about data
def get_opts():
    parser = argparse.ArgumentParser(description='PyTorch implementation of the PSI2.0')
    # about data
    parser.add_argument('--config_file', type=str, default='cfgs/PSI2.0/sgnet_cvae_flow.yaml', help='specify the config file for training')
    parser.add_argument('--dataset', type=str, default='PSI2.0',
                        help='task name: [PSI1.0 | PSI2.0]')
    parser.add_argument('--task_name', type=str, default='ped_traj',
                        help='task name: [ped_intent | ped_traj | driving_decision]')
    parser.add_argument('--video_splits', type=str, default='./*_split.json',
                        help='video splits, [PSI1.0_split | PSI2.0_split | PSI2.0_split_paper]')
    parser.add_argument('--dataset_root_path', type=str, default='psi_dataset',
                        help='Path of the dataset, e.g. frames/video_0001/000.jpg')
    parser.add_argument('--database_path', type=str, default='./database',
                        help='Path of the database created based on the cv_annotations and nlp_annotations')
    parser.add_argument('--database_file', type=str, default=None,
                        help='Filename of the database created based on the cv_annotations and nlp_annotations')
    parser.add_argument('--fps', type=int, default=30,
                        help=' fps of original video, PSI and PEI == 30.')
    parser.add_argument('--seq_overlap_rate', type=float, default=0.9, # 1 means every stride is 1 frame
                        help='Train/Val rate of the overlap frames of slideing windown, (1-rate)*seq_length is the step size')
    parser.add_argument('--test_seq_overlap_rate', type=float, default=1, # 1 means every stride is 1 frame
                        help='Test overlap rate of the overlap frames of slideing windown, (1-rate)*seq_length is the step size')
    parser.add_argument('--intent_num', type=int, default=2,
                        help='Type of intention categories. [2: {cross/not-cross} | 3 {not-sure}]')
    parser.add_argument('--intent_type', type=str, default='mean',
                        help='Type of intention labels, out of 24 annotators. [major | mean | separate | soft_vote];'
                             'only when separate, the nlp reasoning can help, otherwise may take weighted mean of the nlp embeddings')
    parser.add_argument('--observe_length', type=float, default=0,
                        help='Sequence length of one observed clips')
    parser.add_argument('--predict_length', type=float, default=0,
                        help='Sequence length of predicted trajectory/intention')
    parser.add_argument('--max_track_size', type=float, default=60,
                        help='Sequence length of observed + predicted trajectory/intention')
    parser.add_argument('--crop_mode', type=str, default='enlarge',
                        help='Cropping mode of cropping the pedestrian surrounding area')
    parser.add_argument('--balance_data', type=bool, default=False,
                        help='Balance data sampler with randomly class-wise weighted')
    parser.add_argument('--bbox_type', type=str, default='cxcywh',
                        help='Type of bbox. [cxcywh | ltrb]')
    parser.add_argument('--normalize_bbox', type=str, default='zero-one',
                        help='If normalize bbox. [none | zero-one | plus-minus-one]')
    parser.add_argument('--image_shape', type=tuple, default=(1280, 720),
                        help='Image shape: PSI(1280, 720).')
    parser.add_argument('--load_image', type=bool, default=False,
                        help='Do not load image to backbone if not necessary')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path of the pre-trained pt file')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='visualize gt&pred')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train mode')    
    parser.add_argument('--val', action='store_true', default=False,
                        help='val mode')
    parser.add_argument('--trainval', action='store_true', default=False,
                        help='trainval mode')
    parser.add_argument('--test', action='store_true', default=False,
                        help='test mode')
    parser.add_argument('--save_all_checkpoint', action='store_true', default=False,
                        help='save all checkpoint')
    parser.add_argument('--gen_center_of', action='store_true', default=False,
                        help='generate center optical flow')
    
    # about models
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone type [resnet50 | vgg16 | faster_rcnn]')
    parser.add_argument('--freeze_backbone', type=bool, default=False,
                        help='[True | False]')
    parser.add_argument('--relative_bbox_input', action='store_true', default=False,
                        help='Use relative bbox as input')
    
    # about training
    parser.add_argument('--checkpoint_path', type=str, default='./ckpts',
                        help='Path of the stored checkpoints')
    parser.add_argument('--extra_tag', type=str, default=None,
                        help='extra tag of the checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size of dataloader')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers of dataloader')
    parser.add_argument('--resume', type=str, default='',
                        help='ckpt path+filename to be resumed.')
    parser.add_argument('--loss_weights', type=dict, default={'loss_intent': 0.0, 'loss_traj': 1.0, 'loss_driving': 0.0},
                        help='weights of loss terms, {loss_intent, loss_traj}')
    parser.add_argument('--intent_loss', type=list, default=['bce'],
                        help='loss for intent output. [bce | mse | cross_entropy]')
    parser.add_argument('--intent_disagreement', type=float, default=1,
                        help='weather use disagreement to reweight intent loss.threshold to filter training data.'
                             'consensus > 0.5 are selected and reweigh loss; -1.0 means not use; 0.0, means all are used.')
    parser.add_argument('--ignore_uncertain', type=bool, default=False,
                        help='ignore uncertain training samples, based on intent_disagreement')
    parser.add_argument('--intent_positive_weight', type=float, default=0.5,
                        help='weight for intent bce loss: e.g., 0.5 ~= n_neg_class_samples(5118)/n_pos_class_samples(11285)')
    parser.add_argument('--traj_loss', type=list, default=['mse'],
                        help='loss for intent output. [bce | mse | cross_entropy]')
    
    # other parameteres
    parser.add_argument('--val_freq', type=int, default=1,
                        help='frequency of validate')
    parser.add_argument('--test_freq', type=int, default=1,
                        help='frequency of test')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='frequency of print')
    
    args = EasyDict(vars(parser.parse_args()))

    assert args.train or args.val or args.test or args.trainval or args.gen_center_of, "Please specify at least one mode: [train | val | test | gen_center_of]"
    
    with open(args.config_file, 'r') as f:
        try:
             cfg= yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
             cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    
    # args overwrite cfg
    if args.batch_size is not None:
        cfg.optimization.batch_size = args.batch_size
    else:
        args.batch_size = cfg.optimization.batch_size
    if args.epochs is not None:
        cfg.optimization.epochs = args.epochs
    else:
        args.epochs = cfg.optimization.num_epochs
    if cfg.model_cfg.get('flow_module', None) is not None:
        args.use_flow = True
    else:
        args.use_flow = False
    if cfg.model_cfg.get('depth_module', None) is not None:
        args.use_depth = True
    else:
        args.use_depth = False

    args.absolute_bbox_input = not args.relative_bbox_input
    args.observe_length = cfg.model_cfg.observe_length
    args.predict_length = cfg.model_cfg.predict_length
    args.model_name = cfg.model_name

    return args, cfg