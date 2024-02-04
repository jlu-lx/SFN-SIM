import argparse
# import os
parser = argparse.ArgumentParser(description='SFN-SIM')
parser.add_argument('--cuda_name', type=str, default='1')
parser.add_argument('--gpu_ids', type=int, default=1)

parser.add_argument("--writer_name", type=str, default="sfn-sim-writer",
                    help="the name of the writer")

parser.add_argument('--model', type=str, default='SFN_SIM') #
parser.add_argument('--loss_func', type=str, default='SPA_FRE')   # L1, MSE_SSIM ,  SPA_FRE

parser.add_argument('--bio_class', type=str, default='ER',  # F-actin , ER, Microtubules, CCPs
                    help='bio class name')
parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
parser.add_argument('--resume_time', type=str, default='20231124_113724')

parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')

# Test specifications
parser.add_argument('--test_data_dir', type=str, default="../dataset/BioSR/test/ER")   # F-actin , ER, Microtubules, CCPs
parser.add_argument('--test_model_name', type=str, default="SFN_SIM")  

parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--data_path', type=str, default='../dataset/BioSR/train',
                    help='dataset file path')

parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')

parser.add_argument('--num_group', type=int, default=2,
                    help='super resolution scale')

parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--device', default='cuda')

parser.add_argument('--data_train', type=str, default='train',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='test',
                    help='test dataset name')
parser.add_argument('--data_val', type=str, default='val',
                    help='val dataset name')
parser.add_argument('--scale', type=int, default=2,
                    help='super resolution scale')
parser.add_argument('--base_num_every_group', type=int, default=2,
                    help='super resolution scale')

parser.add_argument('--n_colors', type=int, default=9,             # 9 channels,SIM,ER
                    help='number of color channels to use')

parser.add_argument('--augment', action='store_true',
                    help='use data augmentation')
parser.add_argument('--fftloss', action='store_true',
                    help='use data augmentation')
parser.add_argument('--fftd', action='store_true',
                    help='use data augmentation')
parser.add_argument('--fftd_weight', type=float, default=0.1,
                    help='use data augmentation')
parser.add_argument('--fft_weight', type=float, default=0.01)

parser.add_argument('--act', type=str, default='PReLU')
parser.add_argument('--data_range', type=float, default=1)
parser.add_argument('--num_channels', type=int, default=9)   # 单通道
parser.add_argument('--num_features', type=int, default=64)

parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.2,
                    help='residual scaling')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1,
                    help='do test per every N batches')
parser.add_argument('--radius', type=int, default=1,
                    help='the radius of GF')


parser.add_argument('--test_only', action='store_true',# default=True,
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--n_steps', type=int, default=30,
                    help='学习率衰减倍数')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='SPA_FRE',
                    help='loss function configuration')

# Log specifications
parser.add_argument('--root', type=str, default='')

parser.add_argument('--save_path', type=str, default='./experiment',
                    help='file path to save model')

parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--save_name', type=str, default='',
                    help='file name to load')
# parser.add_argument('--resume', type=int, default=0,
#                     help='resume from specific checkpoint')

parser.add_argument('--sche', action='store_true',
                    help='save low-resolution and high-resolution images together')

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
# args.scale = list(map(lambda x: int(x), args.scale.split('+')))


if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

