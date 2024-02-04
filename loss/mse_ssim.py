import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.nn.modules.loss import _Loss




class MSE_SSIM_Loss(_Loss):
    def __init__(self, alpha=1.0, beta=0.1,reduction='mean'):
        super(MSE_SSIM_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        # Normalize y_true and y_pred to [0, 1]

        if y_pred.dim() == 3:
            y_pred = torch.unsqueeze(y_pred,1)  # 从[B,H,W]改为[B,1,H,W]
        
        if y_true.dim() == 3:
            y_true = torch.unsqueeze(y_true,1)  # 从[B,H,W]改为[B,1,H,W]
            
        y_true = y_true.float()
        y_pred = y_pred.float()
        y_true = (y_true - y_true.min()) / (y_true.max() - y_true.min())
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())

        mse_loss = F.mse_loss(y_true, y_pred,reduction=self.reduction)
        ssim_loss = 1 - ssim(y_true, y_pred, data_range=1.0, size_average=True)
        return self.alpha * mse_loss + self.beta * ssim_loss



def loss_mse_ssim(y_true, y_pred):

    if y_pred.dim() == 3:
        y_pred = torch.unsqueeze(y_pred,1)  # 从[B,H,W]改为[B,1,H,W]
        
    ssim_para = 1e-1  # 1e-2
    mse_para = 1

    # normalization
    x = y_true.float()
    y = y_pred.float()
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

    # calculate SSIM and MSE losses
    # print("x shape in ssim before",x.shape)
    # print("y shape in ssim before",y.shape)
    ssim_loss = ssim_para * (1 - ssim(x ,y,data_range=1.0,size_average=True))
    # print("x shape in ssim",x.shape)
    # print("y shape in ssim",y.shape)
    mse_loss = mse_para * F.mse_loss(y, x)

    return mse_loss + ssim_loss
