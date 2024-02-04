from torch import nn
from models import common
from models import fusionBlock
from models.Fourier_Upsampling import fresadd
from option import args


class SIMFreSpaNet(nn.Module):
    def __init__(self, args):
        super(SIMFreSpaNet, self).__init__()
        num_group = args.num_group    # 2 / 3
        print("SFNet num_group: ",num_group)
        self.args = args   
        # Spa Head
        modules_spa_head = [common.ConvBNReLU2D(args.num_channels, out_channels=args.num_features,
                                                kernel_size=3, padding=1, act=args.act)]
        self.spa_head = nn.Sequential(*modules_spa_head)

        # Fre Head
        modules_fre_head = [common.ConvBNReLU2D(args.num_channels, out_channels=args.num_features,
                                                kernel_size=3, padding=1, act=args.act)]
        self.fre_head = nn.Sequential(*modules_fre_head)
        
        # Res Groups
        self.res_groups = nn.ModuleList([ResidualGroup(args) for _ in range(num_group)])

        # tail
        # Spa Tail
        self.spa_tail = nn.Sequential(
            nn.Conv2d(args.num_features, 32, kernel_size=3, padding='same'),
            nn.GELU(),
            fresadd(32),
            nn.Conv2d(32, args.num_features, kernel_size=1),  # New channel adjustment layer
            nn.Conv2d(args.num_features, 1, kernel_size=3, padding='same'),
            nn.Sigmoid(),
        )

        # Fre Tail
        self.fre_tail = nn.Sequential(
            nn.Conv2d(args.num_features, 32, kernel_size=3, padding='same'),
            nn.GELU(),
            fresadd(32),
            nn.Conv2d(32, args.num_features, kernel_size=1),  # New channel adjustment layer
            nn.Conv2d(args.num_features, 1, kernel_size=3, padding='same'),
            nn.Sigmoid(),
        )
        
    def forward(self, x):

        x_spa = self.spa_head(x["lr_up"])
        x_fre = self.fre_head(x["lr_up"])
        
        # Save the original values for long skip connection
        original_x_spa = x_spa
        original_x_fre = x_fre
        
        for g in self.res_groups:
            x_spa,x_fre = g(x_spa,x_fre)
        
        # Apply long skip connection
        x_spa += original_x_spa
        x_fre += original_x_fre     
                   
        spa_out = self.spa_tail(x_spa)
        fre_out = self.fre_tail(x_fre)
        
        return {'img_out': spa_out, 'img_fre': fre_out}


class ResidualGroup(nn.Module):
    def __init__(self, args):
        super(ResidualGroup, self).__init__()
        num_every_group = args.base_num_every_group
        modules_fre = [ common.FreBlock_v2(args.num_features, args),
                        common.FreBlock_v2(args.num_features, args) ]
        modules_spa = [ common.SpaGroup(args.num_features, kernel_size=3, act=args.act, n_resblocks=num_every_group, norm=None)  ]  
        
        self.fre_blocks = nn.Sequential(*modules_fre)
        self.spa_blocks = nn.Sequential(*modules_spa)
    
        ### fusion part 
        self.fuse_block = fusionBlock.FuseBlock_v2(args.num_features)  
        
        
    def forward(self, spa, fre):
        fre_branch = self.fre_blocks(fre)
        spa_branch = self.spa_blocks(spa)
        fuse_feat = self.fuse_block(spa_branch,fre_branch)

        return fuse_feat + spa, fre_branch + fre   # return fuse_feat+res, fre_branch
        

def make_model(args):
    return SIMFreSpaNet(args)
