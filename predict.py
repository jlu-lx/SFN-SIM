import imageio
import os
from utils.metric import prctile_norm, rm_outliers
import numpy as np
import torch.nn as nn
import models
import torch
import datetime
from option import args
import pytz
import glob
from PIL import Image
import cv2
import tifffile

# Define the timezone for China
china_timezone = pytz.timezone('Asia/Shanghai')

# Set the CUDA device name
args.cuda_name = '1'   # "1"
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
    args.model = 'SFN_SIM'  # SFN_SIM model
    bio_class = 'ER'  # Biological class: ER, F-actin, Microtubules, CCPs

    # Set the number of groups based on the biological class
    if bio_class == 'F-actin':
        args.num_group = 3
    elif bio_class == 'Microtubules':
        args.num_group = 3
    else:
        args.num_group = 2

    # Path to the trained models directory
    trained_models_dir = os.path.abspath("./trained_models")

    # Set the test data directory and model path
    args.test_data_dir = f"./**" # Set your test path
    model_path = os.path.join("SFNet-SIMv8", "ER", "result", "epoch239.pth")

    # Define the output directory and file naming convention
    output_name = 'output_' + bio_class + '_' + args.model + '-'
    output_dir = './show_pha_mag' + '/' + output_name

    # Full path to the model
    full_model_path = os.path.join(trained_models_dir, model_path)
    print("full model path: ", full_model_path)
    sd = torch.load(full_model_path, map_location=device)
    model = models.get_model(args)
    
    # Load the model weights and check for missing keys
    missings, _ = model.load_state_dict(sd, strict=True)
    for xx in missings:
        assert 'relative_position_index' in xx or 'attn_mask' in xx, f'essential key {xx} is dropped!'
    print('<All keys matched successfully>')

    # Move the model to the specified device
    model = model.to(args.device)
    
    # Get a list of all image paths in the test data directory
    img_path = glob.glob(args.test_data_dir + '/*')  # level1, level2, ...
    img_path.sort()
    output_dir = output_dir + 'SIM'

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Process each image in the test data directory
    print('Processing ' + args.test_data_dir + '...')
    im_count = 0
    for curp in img_path:
        imgfile = curp
        print("imgfile: ", imgfile)
        img_batch = []
        
        # Read the TIFF file and normalize the slices
        with tifffile.TiffFile(imgfile) as tif:
            if len(tif.pages) != 9:
                print("TIFF NOT 9")
                continue

            img_slices = [prctile_norm(page.asarray()) for page in tif.pages]

        img_batch = img_slices
        
        img = np.array(img_batch)
        img = img[np.newaxis, :, :, :]
        print("img shape: ", img.shape)
        data = {'lr_up': torch.from_numpy(img).float() }

        # Run the model on the input data
        sr = model(to_device(data, device))
        img_out = sr['img_out'].cpu().numpy()
        img_fre = sr['img_fre'].cpu().numpy()
            
        print("image_out shape: ", img_out.shape)
        output = rm_outliers(prctile_norm(np.squeeze(img_out)))
        output_fre = rm_outliers(prctile_norm(np.squeeze(img_fre)))
        print("output", output)
        print("output shape: ", output.shape)
        
        # Save the output images
        outName = curp.replace(args.test_data_dir, output_dir)
        if not outName[-4:] == '.tif':
            outName = outName + '.tif'
        img = Image.fromarray(np.uint16(output * 65535))
        img_fre = Image.fromarray(np.uint16(output_fre * 65535))  # Frequency branch
        im_count = im_count + 1
        img.save(outName)
        img_fre.save("./****" + str(im_count) + ".tif")  # Output path for the frequency branch
        
if __name__ == '__main__':
    main(args)
