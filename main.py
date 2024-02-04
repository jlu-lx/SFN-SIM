import os
from option import args
import torch
import torch.optim as optim
import torch.nn as nn
from data.data_loader import get_data_loader_sim
from torch.utils.tensorboard import SummaryWriter
from utils.metric import img_comp
from utils import util
from loss.mse_ssim import MSE_SSIM_Loss
from loss.fre_loss import AMPLoss, PhaLoss
import glob
import numpy as np
import models
from datetime import datetime
import pytz
import random

# Define the timezone for China
china_timezone = pytz.timezone('Asia/Shanghai')

# Set the CUDA device name
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name

# Function to sort the checkpoint files by epoch number
def sort_key(filename):
    return int(filename.split('epoch')[1].split('.pth')[0])

# Function to count the number of parameters in the model
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# Function to set the random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Set the random seed
set_seed(3407)  # For example, set the seed to 3407

# Number of epochs to train the model
args.epoch = 1000
epochs = args.epochs
model = models.get_model(args)
print(get_parameter_number(model))

# If there are multiple GPUs, use them
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# Initialize the TensorBoard writer
writer = SummaryWriter('./logs/{}'.format(args.writer_name))

# Set the current time for saving models
current_time = datetime.now(china_timezone).strftime('%Y%m%d_%H%M%S')

# Starting epoch number
start_epoch = 0  # Start from the 0th epoch by default

# If resuming training, load the latest checkpoint
if args.resume:
    resume_time = args.resume_time
    current_time = resume_time
    print("current time: ", current_time)
    checkpoint_path = os.path.join(args.save_path, args.writer_name, resume_time, 'model')
    if os.path.exists(checkpoint_path):
        model_checkpoints = sorted(glob.glob(os.path.join(checkpoint_path, '*.pth')), key=sort_key)
        if model_checkpoints:  # Check if there are any checkpoints
            latest_checkpoint = model_checkpoints[-1]
            print(f"Loading checkpoint {latest_checkpoint}")
            model.load_state_dict(torch.load(latest_checkpoint))
            print("load finished!")
            start_epoch = int(os.path.basename(latest_checkpoint).split('epoch')[1].split('.pth')[0])
            print("start epoch: ", start_epoch)

# Get the data loaders for training and validation
train_data_loader, val_data_loader = get_data_loader_sim(args)

# Initialize the optimizer
optimizer = optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)

# Function to move data to the specified device
device = torch.device(args.device)
def to_device(sample, device):
    for key, value in sample.items():
        if key != 'img_name':
            sample[key] = value.to(device, non_blocking=True)
    return sample

# Function to evaluate the model on a dataset
def eval_model(model, dataset, name, epoch, args):
    model.eval()
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'result'), exist_ok=True)
    timer_test = util.timer()
    
    mses, nrmses, psnrs, ssims = [], [], [], []
    for batch, data in enumerate(dataset):
        with torch.no_grad():
            sr = model(to_device(data, device))
            img_gt = data['img_gt'].cpu().numpy()    # Move to CPU
            img_out = sr['img_out'].cpu().numpy()
            mses, nrmses, psnrs, ssims = img_comp(img_gt, img_out, mses, nrmses, psnrs, ssims)  # Calculate various evaluation metrics

    mean_mse = np.mean(mses)
    mean_nrmse = np.mean(nrmses)
    
    mean_psnr = np.mean(psnrs)
    mean_ssim = np.mean(ssims)
    print("Epoch：{}, {}, mse: {:.3f}, nrmse: {:.3f}, psnr: {:.3f}, ssim: {:.3f}".format(epoch+1, name, mean_mse, mean_nrmse, mean_psnr, mean_ssim))
    print('Forward: {:.2f}s\n'.format(timer_test.toc()))
    writer.add_scalar("{}_psnr_DIC".format(name), mean_psnr, epoch)
    writer.add_scalar("{}_ssim_DIC".format(name), mean_ssim, epoch)
    writer.add_scalar("{}_mse_DIC".format(name), mean_mse, epoch)
    writer.add_scalar("{}_nrmse_DIC".format(name), mean_nrmse, epoch)
    
def train_model(model, trainset, epoch, args):
    model.train()
    train_loss = 0
    L1loss = nn.L1Loss().to(device, non_blocking=True)
    mse_ssim_loss = MSE_SSIM_Loss().to(device, non_blocking=True)
    amploss = AMPLoss().to(device, non_blocking=True)
    phaloss = PhaLoss().to(device, non_blocking=True)
    timer_train = util.timer()
    for _, data in enumerate(trainset):
        sr = model(to_device(data, device))      

        if args.loss_func == 'SPA_FRE':                  
            loss = mse_ssim_loss(sr['img_out'], data['img_gt']) + \
                    args.fft_weight * amploss(sr['img_fre'], data['img_gt']) + args.fft_weight * phaloss(
                sr['img_fre'],
                data[
                    'img_gt']) + \
                    mse_ssim_loss(sr['img_fre'], data['img_gt'])
            loss_info = "loss function: mse_ssim and fre loss"
        elif args.loss_func == 'L1':
            loss = L1loss(sr['img_out'], data['img_gt'])   # 
            loss_info = "loss function: L1"

        elif args.loss_func == 'MSE_SSIM':
            loss = mse_ssim_loss(sr['img_out'], data['img_gt'])  #
            loss_info = "loss function: MSE_SSIM"
        else:
            raise NotImplementedError('Loss function [%s] is not implemented' % args.loss_func)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        train_loss = train_loss + loss.item()
        

    cur_lr = optimizer.param_groups[-1]['lr']
        
    print("Epoch：{} loss: {:.3f}".format(epoch+1, train_loss/(len(trainset)) * 255))
    print("loss info: ",loss_info)

    writer.add_scalar('train_loss', train_loss /(len(trainset)) * 255, epoch)
    writer.add_scalar('lr', cur_lr, epoch)

    elapsed_time = timer_train.toc()  # This is your original elapsed time in seconds
    minutes, seconds = divmod(elapsed_time, 60)
    time_now = datetime.now(china_timezone)
    print(f"time: {minutes}m {seconds:.2f}s,  current time: {time_now}, lr: {cur_lr}")    

    os.makedirs(os.path.join(args.save_path, args.writer_name, current_time), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.writer_name, current_time, 'model'), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_path, args.writer_name, current_time, 
                                                'model', 'epoch{}.pth'.format(epoch + 1)))

for i in range(start_epoch, epochs):
    train_model(model, train_data_loader, i, args)
    eval_model(model, val_data_loader, "val", i, args)