import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0')

    # args for dataloader
    parser.add_argument('--is_train', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size_val', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--w', type=int, default=112)
    parser.add_argument('--h', type=int, default=112)
    parser.add_argument('--n_frames_per_clip', type=int, default=32)
    
    
    
    # args for generating the model
    parser.add_argument('--model', type=str, default='resnext')
    parser.add_argument('--arch', type=str, default='resnext-101')
    parser.add_argument('--model_depth', type=int, default=101)
    parser.add_argument('--sample_size', type=int, default=112)
    parser.add_argument('--resnet_shortcut', type=str, default='B')
    parser.add_argument('--resnext_cardinality', type=int, default=32)
    # parser.add_argument('--sample_duration', type=int, default=32)
    parser.add_argument('--pretrain_path', type=str,
    default = 'models/pretrained_models/jester_resnext_101_RGB_32.pth',
    # default='models/pretrained_models/resnext-101-kinetics.pth',
    # default = 'models/pretrained_models/resnet-50-kinetics.pth',
    help='Pretrained path for training model, DO NOT use for testing. Testing trained path is \
    defined in the main script')
    parser.add_argument('--modality', type=str, default='Depth', 
                        help='Modality of input data. RGB, Depth, RGB-D and fusion. Fusion \
                            is only used when testing the two steam model')
    parser.add_argument('--n_classes', type=int, default=83) 
    parser.add_argument('--n_finetune_classes', type=int, default=40)
    parser.add_argument('--ft_begin_index', type=int, default=0, 
                        help='How many parameters need to be fine tuned')
    parser.add_argument('--no_cuda', type=bool, default=False)





    # args for preprocessing
    parser.add_argument('--initial_scale', type=float, default=1,
                        help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int,
                        help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float,
                        help='Scale step for multiscale cropping')
    args = parser.parse_args()
    return args

# def parse_opts():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--is_train', type=bool, default=True)
#     parser.add_argument('--checkpoint', type=str, default='checkpoint_3600.pth')
#     parser.add_argument('--num_workers', type=int, default=8)
#     parser.add_argument('--default_device', type=int, default=1)
#     parser.add_argument('--batch_size_1', type=int, default=256)
#     parser.add_argument('--batch_size_2', type=int, default=1)
#     parser.add_argument('--batch_size_3', type=int, default=256)
#     parser.add_argument('--w', type=int, default=224)
#     parser.add_argument('--h', type=int, default=126)
#     parser.add_argument('--n_frames_per_clip', type=int, default=100)
#     parser.add_argument('--lr_patience', type=int, default=1)
#     args = parser.parse_args()
#     return args