# losses_jittor.py
import jittor as jt
import jittor.nn as nn
from math import exp
import jittor.models as models
L1Loss=nn.L1Loss

# ===================================
# SSIM Loss
# ===================================

def gaussian(window_size, sigma):
    """生成高斯核"""
    gauss = jt.array([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
                      for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """创建 2D 高斯窗口"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = jt.matmul(_1D_window, _1D_window.transpose(1, 0))
    _2D_window = _2D_window.float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """SSIM 计算核心"""
    mu1 = nn.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = nn.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = jt.pow(mu1, 2)
    mu2_sq = jt.pow(mu2, 2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = nn.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = nn.conv2d(
        img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(jt.nn.Module):
    """SSIM Loss - 纯 Jittor 实现"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def execute(self, img1, img2):
        (_, channel, _, _) = img1.shape

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size,
                     channel, self.size_average)


# ===================================
# VGG Loss (纯 Jittor 实现 - 使用 Jittor VGG19)
# ===================================

CONTENT_LAYER = 'relu_16'


# 添加调试代码到 vgg_19_jittor()
def vgg_19_jittor():
    vgg_full = models.vgg19(pretrained=True)
    vgg_features = vgg_full.features
    
    model = nn.Sequential()
    i = 0

    for layer in vgg_features:
        if isinstance(layer, nn.Conv):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
        elif isinstance(layer, nn.Pool):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))
        
        model.add_module(name, layer)
        
        if name == CONTENT_LAYER:
            break
    
    # 验证模型是否为空
    if len(model) == 0:
        raise RuntimeError("VGG 模型为空！")
    
    model.eval()
    model.requires_grad_(False)
    return model

def normalize_batch_jittor(batch):
    """ImageNet 归一化 - Jittor 版本 (按通道处理)"""
    result = batch.clone()
    
    # 通道0 (R)
    result[:, 0:1, :, :] = (result[:, 0:1, :, :] - 0.485) / 0.229
    # 通道1 (G)
    result[:, 1:2, :, :] = (result[:, 1:2, :, :] - 0.456) / 0.224
    # 通道2 (B)
    result[:, 2:3, :, :] = (result[:, 2:3, :, :] - 0.406) / 0.225
    
    return result


class VGGLoss(nn.Module):
    """VGG 感知损失 - 纯 Jittor 实现"""
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.VGG_19 = vgg_19_jittor()
        self.VGG_19.eval()
        self.L1_loss = nn.L1Loss()

    def execute(self, img1, img2):
        # 下采样 0.5x
        img1 = nn.interpolate(img1, scale_factor=0.5, mode="bilinear")
        img2 = nn.interpolate(img2, scale_factor=0.5, mode="bilinear")
        
        # VGG 特征提取
        with jt.no_grad():
            img1_vgg = self.VGG_19(normalize_batch_jittor(img1))
            img2_vgg = self.VGG_19(normalize_batch_jittor(img2))
        
        # 计算 L1 损失
        loss_vgg = self.L1_loss(img1_vgg, img2_vgg)
        return loss_vgg



# ===================================
# GAN Loss (纯 Jittor 实现)
# ===================================

class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        # 修正1: 创建标量，不是列表
        self.real_label_value = target_real_label  # 直接存储 Python 标量
        self.fake_label_value = target_fake_label
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """创建目标张量"""
        if target_is_real:
            target_value = self.real_label_value
        else:
            target_value = self.fake_label_value
        
        # 修正2: 使用 full 直接创建正确形状的张量
        target_tensor = jt.full(prediction.shape, target_value, dtype=prediction.dtype)
        return target_tensor

    def execute(self, prediction, target_is_real):
        """计算损失"""
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss