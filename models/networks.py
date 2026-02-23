import jittor as jt
import jittor.nn as nn
from jittor.nn import init
from collections import OrderedDict
from jittor.lr_scheduler import StepLR

def get_scheduler(optimizer, opt):
    """
    注意: Jittor 的学习率调度器API与PyTorch不同
    """
    if opt.lr_policy == 'step':
        scheduler = StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    else:
        raise NotImplementedError('lr [%s] is not implemented' % opt.lr_policy)
    return scheduler

# 这地方有可能有问题！
def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
    # Jittor 自动使用 CUDA，不需要手动 .to(device)
    # 如果有多 GPU，Jittor 会自动处理
    # if init_type != 'default' and init_type is not None:
    #     init_weights(net, init_type, init_gain=init_gain)
    return net


'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''

def seq(*args):
    if len(args) == 1:
        args = args[0]
    if isinstance(args, nn.Module):
        return args
    modules = OrderedDict()
    if isinstance(args, OrderedDict):
        for k, v in args.items():
            modules[k] = seq(v)
        return nn.Sequential(modules)
    assert isinstance(args, (list, tuple))
    return nn.Sequential(*[seq(i) for i in args])

'''
# ===================================
# Useful blocks
# --------------------------------
# conv (+ normaliation + relu)
# concat
# sum
# resblock (ResBlock)
# resdenseblock (ResidualDenseBlock_5C)
# resinresdenseblock (RRDB)
# ===================================
'''

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
         output_padding=0, dilation=1, groups=1, bias=True,
         padding_mode='zeros', mode='CBR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias))
        elif t == 'X':
            assert in_channels == out_channels
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=in_channels,
                               bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        output_padding=output_padding,
                                        groups=groups,
                                        bias=bias,
                                        dilation=dilation))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'i':
            L.append(nn.InstanceNorm2d(out_channels, affine=False))
        elif t == 'R':
            L.append(nn.ReLU())
        elif t == 'r':
            L.append(nn.ReLU())
        elif t == 'S':
            L.append(nn.Sigmoid())
        elif t == 'P':
            L.append(nn.PReLU())
        elif t == 'L':
            L.append(nn.LeakyReLU(scale=0.1))
        elif t == 'l':
            L.append(nn.LeakyReLU(scale=0.1))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        else:
            raise NotImplementedError('Undefined type: {}'.format(t))
    return seq(*L)


class DWTForward(nn.Conv2d):
    def __init__(self, in_channels=64):
        # 先调用父类初始化
        super(DWTForward, self).__init__(
            in_channels, in_channels*4, 2, 2,
            groups=in_channels, bias=False
        )
        
        # 初始化固定权重
        weight = jt.array(
            [[[[0.5,  0.5], [ 0.5,  0.5]]],
             [[[0.5,  0.5], [-0.5, -0.5]]],
             [[[0.5, -0.5], [ 0.5, -0.5]]],
             [[[0.5, -0.5], [-0.5,  0.5]]]],
            dtype=jt.float32
        ).repeat(in_channels, 1, 1, 1)
        
        # 复制权重数据并冻结整个模块
        self.weight.assign(weight)
        #self.requires_grad_(False)  # 冻结整个模块,与 PyTorch 一致!
        self.weight.requires_grad = False
    


class DWTInverse(nn.ConvTranspose2d):
    def __init__(self, in_channels=64):
        super(DWTInverse, self).__init__(
            in_channels, in_channels//4, 2, 2,
            groups=in_channels//4, bias=False
        )
        
        weight = jt.array(
            [[[[0.5,  0.5], [ 0.5,  0.5]]],
             [[[0.5,  0.5], [-0.5, -0.5]]],
             [[[0.5, -0.5], [ 0.5, -0.5]]],
             [[[0.5, -0.5], [-0.5,  0.5]]]],
            dtype=jt.float32
        ).repeat(in_channels//4, 1, 1, 1)
        self.weight.assign(weight)
        #self.requires_grad_(False)  # 冻结整个模块
        self.weight.requires_grad = False


# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def execute(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y 


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding=padding, bias=bias, mode=mode)

    def execute(self, x):
        res = self.res(x)
        return x + res


# -------------------------------------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding, bias=bias, mode=mode)
        self.ca = CALayer(out_channels, reduction)

    def execute(self, x):
        res = self.res(x)
        res = self.ca(res) 
        return res + x


# -------------------------------------------------------
# Residual Channel Attention Group (RG)
# -------------------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16, nb=12):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]
        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding,
                       bias, mode, reduction) for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        self.rg = nn.Sequential(*RG)

    def execute(self, x):
        res = self.rg(x)
        return res + x