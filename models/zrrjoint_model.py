import jittor as jt
from .base_model import BaseModel
from . import networks as N
import jittor.nn as nn
from . import losses as L
from util.util import get_coord
import numpy as np


class ZRRJOINTModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        super(ZRRJOINTModel, self).__init__(opt)

        self.opt = opt
        self.loss_names = ['GCMModel_L1', 'LiteISPNet_L1', 'LiteISPNet_SSIM', 'LiteISPNet_VGG', 'Total']
        self.visual_names = ['data_dslr', 'data_out', 'GCMModel_out']  # 去掉 dslr_warp, dslr_mask
        self.model_names = ['LiteISPNet', 'GCMModel'] 
        self.optimizer_names = ['LiteISPNet_optimizer_%s' % opt.optimizer,
                                'GCMModel_optimizer_%s' % opt.optimizer]

        isp = LiteISPNet(opt)
        self.netLiteISPNet = N.init_net(isp, opt.init_type, opt.init_gain, opt.gpu_ids)

        gcm = GCMModel(opt)
        self.netGCMModel = N.init_net(gcm, opt.init_type, opt.init_gain, opt.gpu_ids)


        if self.isTrain:        
            # Jittor 优化器
            self.optimizer_LiteISPNet = jt.optim.Adam(
                self.netLiteISPNet.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, opt.beta2),
                weight_decay=opt.weight_decay
            )
            self.optimizer_GCMModel = jt.optim.Adam(
                self.netGCMModel.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, opt.beta2),
                weight_decay=opt.weight_decay
            )
            self.optimizers = [self.optimizer_LiteISPNet, self.optimizer_GCMModel]

            # Jittor 损失函数
            self.criterionL1 = L.L1Loss()
            self.criterionSSIM = L.SSIMLoss()
            self.criterionVGG = L.VGGLoss()

        self.data_ispnet_coord = {}

    def set_input(self, input):
        """输入数据转换为 Jittor 变量"""
        self.data_raw = jt.array(input['raw'])
        self.data_raw_demosaic = jt.array(input['raw_demosaic'])
        self.data_dslr = jt.array(input['dslr'])
        self.data_gcm_coord = jt.array(input['coord'])
        self.image_paths = input['fname']

    def forward(self):
        self.GCMModel_out = self.netGCMModel(
            self.data_raw_demosaic, 
            self.data_dslr, 
            self.data_gcm_coord
        )
        
        N, C, H, W = self.data_raw.shape
        index = str(self.data_raw.shape)
        if index not in self.data_ispnet_coord:
            if self.opt.pre_ispnet_coord:
                data_ispnet_coord = get_coord(H=H, W=W)
            else:
                data_ispnet_coord = get_coord(H=H, W=W, x=1, y=1)
            data_ispnet_coord = np.expand_dims(data_ispnet_coord, axis=0)
            data_ispnet_coord = np.tile(data_ispnet_coord, (N, 1, 1, 1))
            self.data_ispnet_coord[index] = jt.array(data_ispnet_coord)

        self.data_out = self.netLiteISPNet(
            self.data_raw, 
            self.data_ispnet_coord[index]
        )

    def backward(self):

        
        self.loss_GCMModel_L1 = self.criterionL1(
            self.GCMModel_out, self.data_dslr
        ).mean()
        
        self.loss_LiteISPNet_L1 = self.criterionL1(
            self.data_out, self.data_dslr
        ).mean()
        
        self.loss_LiteISPNet_SSIM = 1 - self.criterionSSIM(
            self.data_out, self.data_dslr
        ).mean()
        
        self.loss_LiteISPNet_VGG = self.criterionVGG(
            self.data_out, self.data_dslr
        ).mean()
        
        # 总损失
        self.loss_Total = (
            self.loss_GCMModel_L1 + 
            self.loss_LiteISPNet_L1 + 
            self.loss_LiteISPNet_VGG + 
            self.loss_LiteISPNet_SSIM * 0.15
        )
        
        # Jittor 反向传播
        self.optimizer_LiteISPNet.backward(self.loss_Total)
        self.optimizer_GCMModel.backward(self.loss_Total)

    def optimize_parameters(self):
        """优化参数 - Jittor 版本"""
        self.forward()
        self.optimizer_LiteISPNet.zero_grad()
        self.optimizer_GCMModel.zero_grad()
        self.backward()
        self.optimizer_LiteISPNet.step()
        self.optimizer_GCMModel.step()


# ============================================================
# GCMModel - Jittor 版本 (保持不变)
# ============================================================

class GCMModel(nn.Module):
    def __init__(self, opt):
        super(GCMModel, self).__init__()
        self.opt = opt
        self.ch_1 = 32
        self.ch_2 = 64
        guide_input_channels = 8
        align_input_channels = 5
        self.gcm_coord = opt.gcm_coord

        if not self.gcm_coord:
            guide_input_channels = 6
            align_input_channels = 3
        
        self.guide_net = nn.Sequential(
            nn.Sequential(
                nn.Conv(guide_input_channels, self.ch_1, 7, stride=2, padding=0),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv(self.ch_1, self.ch_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv(self.ch_1, self.ch_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv(self.ch_1, self.ch_2, 1, stride=1, padding=0)
        )

        self.align_head = nn.Sequential(
            nn.Conv(align_input_channels, self.ch_2, 1, padding=0),
            nn.ReLU()
        )

        self.align_base = nn.Sequential(
            nn.Conv(self.ch_2, self.ch_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv(self.ch_2, self.ch_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv(self.ch_2, self.ch_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        
        self.align_tail = nn.Conv(self.ch_2, 3, 1, padding=0)

    def execute(self, demosaic_raw, dslr, coord=None):
        """前向传播 - Jittor 版本"""
        demosaic_raw = jt.pow(demosaic_raw, 1/2.2)
        
        if self.gcm_coord:
            guide_input = jt.cat((demosaic_raw, dslr, coord), 1)
            base_input = jt.cat((demosaic_raw, coord), 1)
        else:
            guide_input = jt.cat((demosaic_raw, dslr), 1)
            base_input = demosaic_raw

        guide = self.guide_net(guide_input)
    
        out = self.align_head(base_input)
        out = guide * out + out
        out = self.align_base(out)
        out = self.align_tail(out) + demosaic_raw
        
        return out


# ============================================================
# LiteISPNet - Jittor 版本 (保持不变)
# ============================================================

class LiteISPNet(nn.Module):
    def __init__(self, opt):
        super(LiteISPNet, self).__init__()
        self.opt = opt
        ch_1 = 64
        ch_2 = 128
        ch_3 = 128
        n_blocks = 4
        self.pre_ispnet_coord = opt.pre_ispnet_coord

        self.head = nn.Sequential(
            nn.Conv(4, ch_1, 3, padding=1)
        )
        # self.head = nn.conv(4, ch_1, 3, padding=1)

        if self.pre_ispnet_coord:
            self.pre_coord = PreCoord(pre_train=True)

        self.down1 = nn.Sequential(
            nn.Conv(ch_1+2, ch_1+2, 3, padding=1),
            N.RCAGroup(in_channels=ch_1+2, out_channels=ch_1+2, nb=n_blocks),
            nn.Conv(ch_1+2, ch_1, 3, padding=1),
            N.DWTForward(ch_1)
        )

        self.down2 = nn.Sequential(
            nn.Conv(ch_1*4, ch_1, 3, padding=1),
            N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            N.DWTForward(ch_1)
        )

        self.down3 = nn.Sequential(
            nn.Conv(ch_1*4, ch_2, 3, padding=1),
            N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
            N.DWTForward(ch_2)
        )

        self.middle = nn.Sequential(
            nn.Conv(ch_2*4, ch_3, 3, padding=1),
            N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
            N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
            nn.Conv(ch_3, ch_2*4, 3, padding=1)
        )

        self.up3 = nn.Sequential(
            N.DWTInverse(ch_2*4),
            N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
            nn.Conv(ch_2, ch_1*4, 3, padding=1)
        )

        self.up2 = nn.Sequential(
            N.DWTInverse(ch_1*4),
            N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            nn.Conv(ch_1, ch_1*4, 3, padding=1)
        )

        self.up1 = nn.Sequential(
            N.DWTInverse(ch_1*4),
            N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
            nn.Conv(ch_1, ch_1, 3, padding=1)
        )

        self.tail = nn.Sequential(
            nn.Conv(ch_1, ch_1*4, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv(ch_1, 3, 3, padding=1)
        )

    def execute(self, raw, coord=None):
        """前向传播 - Jittor 版本"""
        input = jt.pow(raw, 1/2.2)
        h = self.head(input)
        
        if self.pre_ispnet_coord:
            coord_out = self.pre_coord(raw)
            coord_out = coord_out.view(coord_out.shape[0], 2, 1, 1)
            coord = coord * coord_out
            h_coord = jt.cat([h, coord], dim=1)
        else:
            h_coord = jt.cat([h, coord], dim=1)
        
        d1 = self.down1(h_coord)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        m = self.middle(d3) + d3
        u3 = self.up3(m) + d2
        u2 = self.up2(u3) + d1
        u1 = self.up1(u2) + h
        out = self.tail(u1)

        return out


# ============================================================
# PreCoord - Jittor 版本
# ============================================================

class PreCoord(nn.Module):
    def __init__(self, pre_train=True):
        super(PreCoord, self).__init__()
        self.ch_1 = 64
        
        # 224 -> 111 -> 54 -> 26 -> 13
        self.down = nn.Sequential(
            nn.Sequential(  # down.0
                nn.Conv(4, self.ch_1, 3, stride=2, padding=0),
                nn.ReLU()
            ),
            nn.Sequential(  # down.1
                nn.Conv(self.ch_1, self.ch_1, 3, stride=2, padding=0),
                nn.ReLU()
            ),
            nn.Sequential(  # down.2
                nn.Conv(self.ch_1, self.ch_1, 3, stride=2, padding=0),
                nn.ReLU()
            ),
            nn.Sequential(  # down.3
                nn.Conv(self.ch_1, self.ch_1, 3, stride=2, padding=0),
                nn.ReLU()
            )
        )
        
        # fc 层: 13×13×64=10816 -> 256 -> 2
        self.fc = nn.Sequential(
            nn.Linear(self.ch_1*13*13, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
        if pre_train:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """从 PyTorch 权重文件加载到 Jittor 模型"""
        import torch
        import os
        
        weight_path = './ckpt/coord.pth'
        if not os.path.exists(weight_path):
            print(f"Warning: PreCoord权重文件不存在: {weight_path}")
            return
        
        torch_state = torch.load(weight_path, map_location='cpu')['state_dict']
        jittor_state = {}
        for k, v in torch_state.items():
            jittor_state[k] = jt.array(v.numpy())
        
        self.load_state_dict(jittor_state)
        print(f"✓ 成功加载 PreCoord 预训练权重: {weight_path}")

    def execute(self, raw):
        """前向传播 - Jittor 版本"""
        N, C, H, W = raw.shape
        input = raw
        
        if H != 224 or W != 224:
            input = nn.interpolate(
                input, 
                size=[224, 224], 
                mode='bilinear', 
                align_corners=True
            )
        
        down = self.down(input)
        down = down.view(N, self.ch_1*13*13)
        out = self.fc(down)
        
        return out