
import math
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio



def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y

def quantize(img):
    return (img *  255).clip(0, 255).round()

def tensor2img(tensor):
    return quantize(tensor.detach().cpu().numpy()).astype(np.uint8).transpose(0, 2, 3, 1)

def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    #
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    mse = np.mean((img1_np - img2_np)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def torch_psnr(img_pred, img_gt):
    img_gt = tensor2img(img_gt)
    img_pred = tensor2img(img_pred)

    img_pred, img_gt = img_pred[:, 8: -8, 8:-8, :], img_gt[:, 8: -8, 8:-8, :]

    sum_psnr = []
    sum_ssim = []
    for i in range(img_gt.shape[0]):
        sum_psnr.append(calc_psnr(rgb2ycbcr(img_pred[i]), rgb2ycbcr(img_gt[i])))
        sum_ssim.append(structural_similarity(rgb2ycbcr(img_pred[i]), rgb2ycbcr(img_gt[i]), data_range=255))
    return np.mean(sum_psnr),  np.mean(sum_ssim)





def img_comp(gt, pr, mses=None, nrmses=None, psnrs=None, ssims=None):
    if ssims is None:
        ssims = []
    if psnrs is None:
        psnrs = []
    if nrmses is None:
        nrmses = []
    if mses is None:
        mses = []
    gt, pr = np.squeeze(gt), np.squeeze(pr)

    if gt.ndim == 2:
        n = 1
        gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
        pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
    else:
        n = np.size(gt, 0)

    for i in range(n):
        mses.append(mean_squared_error(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
        nrmses.append(normalized_root_mse(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
        psnrs.append(peak_signal_noise_ratio(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
        ssims.append(structural_similarity(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i])),data_range=1.0))
    return mses, nrmses, psnrs, ssims



def diffxy(img, order=3):
    for _ in range(order):
        img = prctile_norm(img)
        d = np.zeros_like(img)
        dx = (img[1:-1, 0:-2] + img[1:-1, 2:]) / 2
        dy = (img[0:-2, 1:-1] + img[2:, 1:-1]) / 2
        d[1:-1, 1:-1] = img[1:-1, 1:-1] - (dx + dy) / 2
        d[d < 0] = 0
        img = d
    return img


def rm_outliers(img, order=3, thresh=0.2):
    img_diff = diffxy(img, order)
    mask = img_diff > thresh
    img_rm_outliers = img
    img_mean = np.zeros_like(img)
    for i in [-1, 1]:
        for a in range(0, 2):
            img_mean = img_mean + np.roll(img, i, axis=a)
    img_mean = img_mean / 4
    img_rm_outliers[mask] = img_mean[mask]
    img_rm_outliers = prctile_norm(img_rm_outliers)
    return img_rm_outliers