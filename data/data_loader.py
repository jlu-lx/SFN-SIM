import torch
import cv2
import numpy as np
import glob
import os
# from torch.utils.data import RandomSampler
from torch.utils.data import Dataset, DataLoader
import imageio

def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


    def __init__(self, images_path, data_path, gt_path,
                 norm_flag=1, resize_flag=0, scale=2, bn=0):
        # 传入必要参数，包括图像路径、高度、宽度等
        self.images_path = images_path
        self.data_path = data_path
        self.gt_path = gt_path
        self.norm_flag = norm_flag
        self.resize_flag = resize_flag
        self.scale = scale
        self.bn = bn

    def __len__(self):
        # 返回图像数目
        return len(self.images_path)

    def __getitem__(self, idx):
        path = self.images_path[idx]
        
        filename = os.path.basename(path)
        # 读取单张图像
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # 若需要调整图像大小，则按比例进行缩放
        if self.resize_flag == 1:
            img = cv2.resize(img, (self.height*self.scale, self.width*self.scale))

        # 读取对应标签图像
        path_gt = path.replace(self.data_path, self.gt_path)
        
        gt = cv2.imread(path_gt, cv2.IMREAD_UNCHANGED)

        # 根据 norm_flag 参数选择是否进行像素值归一化
        if self.norm_flag:
            img = prctile_norm(img)   # prctile_norm()函数可能只接受2维的，扩展维度应该在它后面
            gt = prctile_norm(gt)
        else:
            img = img / 65535.    # 最大像素值为 65535
            gt = gt / 65535.

        # 根据 bn 参数选择是否进行像素值平移和缩放（做 BatchNormalization）
        if self.bn:
            img = (img - 0.5) / 0.5
            gt = (gt - 0.5) / 0.5

        # 当输入为单通道图像时，需要将通道维度扩展
        if len(img.shape) == 2:             
            img = img[:,:,np.newaxis]       
        if len(gt.shape) == 2:             
            gt = gt[:,:,np.newaxis]         

        # 图像数据格式转换，将维度顺序变成 (1, height, width)
        img = img.transpose((2, 0, 1))  
        gt = gt.transpose((2, 0, 1))    
        img = torch.from_numpy(img)     
        gt = torch.from_numpy(gt)

        # 转换数据类型为 float，并返回 tensor 形式的结果
        # return img.float(), img.float()
    
        return {'lr_up': img.float(), 'img_gt': gt.float(), 'img_name': filename}

# 定义 Multi_Channel_Dataset 类，用于读取并处理多通道图像数据
class Multi_Channel_Dataset(Dataset):
    def __init__(self, images_path, data_path, gt_path, norm_flag=1, resize_flag=0, scale=2, wf=0):
        # 传入必要参数，包括图像路径、高度、宽度等
        self.images_path = images_path
        self.data_path = data_path
        self.gt_path = gt_path
        self.norm_flag = norm_flag
        self.resize_flag = resize_flag
        self.scale = scale
        self.wf = wf

    def __getitem__(self, index):
        path = self.images_path[index]
        # 获取多通道图像文件夹下的所有 tif 格式图像
        # print("path: ",path)
        filename = os.path.basename(path)
        img_path = glob.glob(path + '/*.tif')
        img_path.sort()
        cur_img = []
        for cur in img_path:
            # 逐个读取每张图像
            img = imageio.imread(cur).astype(np.float32)
            if self.resize_flag == 1:
                img = cv2.resize(img, (self.height * self.scale, self.width * self.scale))

            if self.norm_flag:
                # 根据 norm_flag 参数选择是否进行像素值归一化
                img = prctile_norm(img)
            cur_img.append(img)

        path_gt = path.replace(self.data_path, self.gt_path) + '.tif'
        cur_gt = imageio.imread(path_gt).astype(np.float32)

        # 根据 norm_flag 参数选择是否进行像素值归一化
        if self.norm_flag:
            cur_gt = prctile_norm(cur_gt)
        else:
            cur_img = np.array(cur_img) / 65535.
            cur_gt = cur_gt / 65535.
        cur_img = np.array(cur_img)

        # 将标签图像转换为 tensor 格式
        # print("cur_gt shape before: ",cur_gt.shape)      # (256x256)
        cur_gt = torch.from_numpy(cur_gt).unsqueeze(0)   # [1,256,256]
        # print("cur_gt shape after: ",cur_gt.shape)
        cur_img = torch.from_numpy(cur_img)
        
        # return cur_img.float(), cur_gt.float()
        return {'lr_up': cur_img.float(), 'img_gt': cur_gt.float(), 'img_name': filename}
    
    def __len__(self):
        # 返回图像数目
        return len(self.images_path)

class Multi_Channel_Dataset_test(Dataset):
    def __init__(self, images_path, data_path, gt_path, 
                 norm_flag=1, resize_flag=0, scale=2, bn=0):
        # 传入必要参数，包括图像路径、高度、宽度等
        self.images_path = images_path
        self.data_path = data_path
        self.gt_path = gt_path
        self.norm_flag = norm_flag
        self.resize_flag = resize_flag
        self.scale = scale
        self.bn = bn

    def __len__(self):
        # 返回图像数目
        return len(self.images_path)

    def __getitem__(self, idx):
        path = self.images_path[idx]
        
        filename = os.path.basename(path)

        # 使用 imageio 来读取图像，因为它支持多通道的 tif 文件
        img = imageio.v2.imread(path).astype(np.float32)
        channels = img.shape[0]   # assuming channels is the first dimension
        # print("channels: ",channels)
        # 读取对应标签图像
        path_gt = path.replace(self.data_path, self.gt_path)
        gt = imageio.imread(path_gt).astype(np.float32)
        
        # 根据 norm_flag 参数选择是否进行像素值归一化
        if self.norm_flag:
            for i in range(channels):
                img[i] = prctile_norm(img[i])
            gt = prctile_norm(gt)
        else:
            img = img / 65535.    # 最大像素值为 65535
            gt = gt / 65535.
            
        gt = torch.from_numpy(gt).unsqueeze(0)   # [1,512,512]
        img = torch.from_numpy(img)     
    
        return {'lr_up': img.float(), 'img_gt': gt.float(), 'img_name': filename}

# 定义 data_loader_multi_channel 函数，用于创建多通道图像的数据加载器
def data_loader_multi_channel(images_path, data_path, gt_path, batch_size, shuffle=True,num_workers=16, norm_flag=1, resize_flag=0, scale=2, wf=0):
    # 定义 Multi_Channel_Dataset 类的实例化对象
    dataset = Multi_Channel_Dataset(images_path, data_path, gt_path, norm_flag, resize_flag, scale, wf)
    # 创建并返回数据加载器对象
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return dataloader


# 定义 data_loader_multi_channel 函数，用于创建多通道图像的数据加载器
def data_loader_multi_channel_test(images_path, data_path, gt_path, batch_size=1, shuffle=False,num_workers=16, norm_flag=1, resize_flag=0, scale=2, wf=0):
    # 定义 Multi_Channel_Dataset 类的实例化对象
    dataset = Multi_Channel_Dataset_test(images_path, data_path, gt_path, norm_flag, resize_flag, scale, wf)
    # 创建并返回数据加载器对象
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return dataloader

# SIM
def get_data_loader_sim(args):
    data_loader = data_loader_multi_channel           # SIM dataloader
    train_path = os.path.join(args.data_path, args.bio_class, 'training')

    print("train_images_path:",train_path)
    validate_path = os.path.join(args.data_path, args.bio_class, 'validate')
    train_gt_path = os.path.join(args.data_path, args.bio_class, 'training_gt')
    validate_gt_path = os.path.join(args.data_path, args.bio_class, 'validate_gt')

    # 训练时路径列表
    train_path_list = glob.glob(train_path + '/*')

    train_data_loader = data_loader(train_path_list, train_path, train_gt_path,
                                        batch_size=args.batch_size,shuffle=True, num_workers=24)

    # val时路径列表
    val_path_list = glob.glob(validate_path + '/*')

    val_data_loader = data_loader(val_path_list, validate_path, validate_gt_path,
                                       batch_size=1, shuffle=False, num_workers=1)
                                           
    return train_data_loader,val_data_loader

# 模型训练完成后，对模型测试的函数,返回一个test_loaders列表，包含各个level的dataloader
def build_test_dataset_sim(args):

    test_path = args.test_data_dir + '/test/'
    test_gt_path = args.test_data_dir + '/test_gt/'

    test_path_list = glob.glob(test_path + '/*')
    test_path_list = sorted(test_path_list)   # # level01, level02, level03, level04..
    print(test_path_list)
    test_path_dict = {}
    for i in test_path_list:
        test_path_dict[os.path.basename(i)] = glob.glob(i + '/*')

    test_loaders = []                       # level01, level02, level03, level04...
    
    # print("test_path_dict: ",test_path_dict.keys())
    # 从test_path_dict中取出每个值,并且构建成dataloader,添加到test_loaders中
    for level_path in test_path_dict.values():
        test_loader = data_loader_multi_channel_test(images_path=level_path,data_path=test_path,gt_path=test_gt_path
                                              ,batch_size=1,shuffle=False)
        test_loaders.append(test_loader)

    print("len(test_loaders):",len(test_loaders))
 
    return test_loaders

