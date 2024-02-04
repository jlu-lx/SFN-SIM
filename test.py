from data.data_loader import build_test_dataset_sim
import os
from utils.metric import img_comp
import numpy as np
import torch.nn as nn
import models
import torch
import datetime
from option import args
import pytz

# Define the timezone for China
china_timezone = pytz.timezone('Asia/Shanghai')

# Set the CUDA device name
args.cuda_name = '1'   # "1"

# Set the environment variable to use the specified CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name

# Define the device to use for computation (CPU or GPU)
device = torch.device(args.device)

# Function to move data to the specified device
def to_device(sample, device):
    for key, value in sample.items():
        if key != 'img_name':
            sample[key] = value.to(device, non_blocking=True)
    return sample

# Decorator to disable gradient computation for the main function
@torch.no_grad()
def main(args):
    args.model = 'SFN_SIM'
    args.test_data_dir = '../dataset/BioSR/test/ER'
    trained_models_dir = os.path.abspath("./trained_models")
    model_path = os.path.join("*", "*", "*.pth") # Set the path for the pretrained model 

    args.test_data_dir = "../dataset/BioSR/test/ER"

    args.num_group = 2

    full_model_path = os.path.join(trained_models_dir, model_path)
    print("full model path: ",full_model_path)
    sd = torch.load(full_model_path,map_location=device)
    model = models.get_model(args)

    # Load the model weights and check for missing keys
    missings,_ = model.load_state_dict(sd, strict=True)
    for xx in missings:
        assert 'relative_position_index' in xx or 'attn_mask' in xx, f'essential key {xx} is dropped!'
    print('<All keys matched successfully>')

    model = model.to(args.device)
    test_loaders = build_test_dataset_sim(args)   # Build the test dataset for SIM reconstruction (9)
    
    # Lists to store the mean values of various metrics
    mses_mean, nrmses_mean, psnrs_mean, ssims_mean = [], [], [], []
    model.eval()                                          
    for test_loader in test_loaders:                        # Iterate through the test loaders for different levels
        mses, nrmses, psnrs, ssims = [], [], [], []         # Initialize lists for metrics at the current level
        for batch, data in enumerate(test_loader):        # Iterate through the batches in the current test loader
            with torch.no_grad():
                sr = model(to_device(data, device))
                img_gt = data['img_gt'].cpu().numpy()       # Move the ground truth image to CPU and convert to numpy array
                img_out = sr['img_out'].cpu().numpy()     # Move the output image to CPU and convert to numpy array
                
                mses, nrmses, psnrs, ssims = img_comp(img_gt, img_out, mses, nrmses, psnrs, ssims)  # Calculate various evaluation metrics

        # Append the mean values of metrics for the current level to the respective lists
        mses_mean.append(round(np.mean(mses), 3))
        nrmses_mean.append(round(np.mean(nrmses), 3))
        psnrs_mean.append(round(np.mean(psnrs), 3))
        ssims_mean.append(round(np.mean(ssims), 3))

    # Print the rounded mean values of the metrics
    print('mses_mean:', mses_mean)
    print('nrmses_mean:', nrmses_mean)
    print('psnrs_mean:', psnrs_mean)
    print('ssims_mean:', ssims_mean)

    # Get the current time in the China timezone
    now = datetime.datetime.now(china_timezone).strftime('%Y-%m-%d %H:%M:%S')
    with open(f'./metrics/SFNet/metrics_SIM.txt', 'a') as f:
        f.writelines(now + '\n')
        f.writelines(f'test on- {model_path}\n\n')
        for level, (mse, nrmse, psnr, ssim) in enumerate(zip(mses_mean, nrmses_mean, psnrs_mean, ssims_mean), start=1):
            f.writelines(f'Level {level}:\n')
            f.writelines(f'MSE: {mse}\n')
            f.writelines(f'NRMSE: {nrmse}\n')
            f.writelines(f'PSNR: {psnr}\n')
            f.writelines(f'SSIM: {ssim}\n\n')

            
if __name__ == '__main__':
    main(args)
