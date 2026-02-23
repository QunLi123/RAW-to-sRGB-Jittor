# base_model_jittor.py
import os
import jittor as jt
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks as networks
import torch  # 保留用于 PWC-Net
import torch.nn.functional as TF
from util.util import jt_save
import math 
import jittor.nn as nn
import numpy as np

# zrr和srraw的基类
class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.scale = opt.scale

        # Jittor 自动管理 GPU，不需要手动指定 device
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.optimizer_names = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.start_epoch = 0
                
        self.backwarp_tenGrid = {}
        self.backwarp_tenPartial = {}

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, opt=None):
        opt = opt if opt is not None else self.opt
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) \
                               for optimizer in self.optimizers]
            for scheduler in self.schedulers:
                scheduler.last_epoch = opt.load_iter
        if opt.load_iter > 0 or opt.load_path != '':
            load_suffix = opt.load_iter
            self.load_networks(load_suffix)
            if opt.load_optimizers:
                self.load_optimizers(opt.load_iter)

        self.print_networks(opt.verbose)

    def eval(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.eval()

    def train(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.train()

    def test(self):
        with jt.no_grad():
            self.forward()

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        for i, scheduler in enumerate(self.schedulers):
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
            # Jittor: scheduler.step() 会自动更新 optimizer.lr
            print('lr of %s = %.7f' % (
                self.optimizer_names[i], self.optimizers[i].lr))

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if 'xy' in name or 'coord' in name:
                visual_ret[name] = getattr(self, name).detach()
            else:
                visual_ret[name] = jt.clamp(
                            getattr(self, name).detach()*255, 0, 255).round()
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            save_filename = '%s_model_%d.pth' % (name, epoch)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, 'net' + name)
            # Jittor 模型直接保存 state_dict
            state = {'state_dict': net.state_dict()}
            jt_save(state, save_path)
        self.save_optimizers(epoch)

    def load_networks(self, epoch):
        import pickle
        for name in self.model_names:
            load_filename = '%s_model_%d.pth' % (name, epoch)
            if self.opt.load_path != '':
                load_path = self.opt.load_path
            else:
                load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            
            # 使用 pickle 加载
            with open(load_path, 'rb') as f:
                state_dict = pickle.load(f)
            print('loading the model from %s' % (load_path))
            
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net_state = net.state_dict()
            is_loaded = {n:False for n in net_state.keys()}
            for name, param in state_dict['state_dict'].items():
                if name in net_state:
                    try:
                        # 从 numpy 转回 Jittor Var
                        if isinstance(param, np.ndarray):
                            net_state[name].assign(jt.array(param))
                        else:
                            net_state[name].assign(param)
                        is_loaded[name] = True
                    except Exception:
                        print('While copying the parameter named [%s], '
                            'whose dimensions in the model are %s and '
                            'whose dimensions in the checkpoint are %s.'
                            % (name, list(net_state[name].shape),
                                list(param.shape if hasattr(param, 'shape') else 'unknown')))
                        raise RuntimeError
                else:
                    print('Saved parameter named [%s] is skipped' % name)
            mark = True
            for name in is_loaded:
                if not is_loaded[name]:
                    print('Parameter named [%s] is randomly initialized' % name)
                    mark = False
            if mark:
                print('All parameters are initialized using [%s]' % load_path)

            self.start_epoch = epoch

    def save_optimizers(self, epoch):
        assert len(self.optimizers) == len(self.optimizer_names)
        for id, optimizer in enumerate(self.optimizers):
            save_filename = self.optimizer_names[id]
            state = {'name': save_filename,
                     'epoch': epoch,
                     'state_dict': optimizer.state_dict()}
            save_path = os.path.join(self.save_dir, save_filename+'.pth')
            jt_save(state, save_path)

    def load_optimizers(self, epoch):
        assert len(self.optimizers) == len(self.optimizer_names)
        for id, optimizer in enumerate(self.optimizer_names):
            load_filename = self.optimizer_names[id]
            load_path = os.path.join(self.save_dir, load_filename+'.pth')
            print('loading the optimizer from %s' % load_path)
            state_dict = jt.load(load_path)
            assert optimizer == state_dict['name']
            assert epoch == state_dict['epoch']
            self.optimizers[id].load_state_dict(state_dict['state_dict'])

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M'
                      % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # ===================================================================
    # 关键！PWC-Net 相关方法：Jittor ↔ PyTorch 边界转换
    # ===================================================================
    
    def estimate(self, tenFirst, tenSecond, net):
        """
        PWC-Net 光流估计 - 边界转换版本
        输入: Jittor tensors
        输出: Jittor tensor
        中间: 使用 PyTorch PWC-Net
        """
        assert(tenFirst.shape[3] == tenSecond.shape[3])
        assert(tenFirst.shape[2] == tenSecond.shape[2])
        intWidth = tenFirst.shape[3]
        intHeight = tenFirst.shape[2]

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        # Jittor 插值
        tenPreprocessedFirst = nn.interpolate(
            tenFirst, 
                                size=(intPreprocessedHeight, intPreprocessedWidth), 
                                mode='bilinear', align_corners=False)
        tenPreprocessedSecond = nn.interpolate(
            tenSecond, 
                                size=(intPreprocessedHeight, intPreprocessedWidth), 
                                mode='bilinear', align_corners=False)

        # ============ 边界转换：Jittor → PyTorch ============
        torch_first = torch.from_numpy(tenPreprocessedFirst.numpy()).cuda()
        torch_second = torch.from_numpy(tenPreprocessedSecond.numpy()).cuda()
        
        # PyTorch PWC-Net 推理
        with torch.no_grad():
            torch_flow = net(torch_first, torch_second)
        
        # ============ 边界转换：PyTorch → Jittor ============
        jt_flow = jt.array(torch_flow.cpu().numpy())
        
        # Jittor 插值回原尺寸
        tenFlow = 20.0 * nn.interpolate(
            jt_flow, 
                         size=(intHeight, intWidth), 
                         mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow[:, :, :, :]
    
    def backwarp(self, tenInput, tenFlow):
        """Jittor 版本的 backwarp（光流 warp）"""
        index = str(tenFlow.shape)
        if index not in self.backwarp_tenGrid:
            tenHor = jt.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 
                                 1.0 - (1.0 / tenFlow.shape[3]), 
                                 tenFlow.shape[3]).view(1, 1, 1, -1).expand(
                                     -1, -1, tenFlow.shape[2], -1)
            tenVer = jt.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 
                                 1.0 - (1.0 / tenFlow.shape[2]), 
                                 tenFlow.shape[2]).view(1, 1, -1, 1).expand(
                                     -1, -1, -1, tenFlow.shape[3])
            self.backwarp_tenGrid[index] = jt.cat([tenHor, tenVer], 1)

        if index not in self.backwarp_tenPartial:
            # 修正：使用 tenFlow.new_ones() 遵循官方API
            self.backwarp_tenPartial[index] = tenFlow.new_ones([
                 tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])

        # 修正：使用 jt.cat 保持与 PyTorch 一致
        tenFlow = jt.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
                          tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
        tenInput = jt.cat([tenInput, self.backwarp_tenPartial[index]], 1)

        tenOutput = nn.grid_sample(input=tenInput, 
                    grid=(self.backwarp_tenGrid[index] + tenFlow).permute(0, 2, 3, 1), 
                    mode='bilinear', padding_mode='zeros', align_corners=False)

        return tenOutput

    def get_backwarp(self, tenFirst, tenSecond, net, flow=None):
        """获取 warped 图像和 mask"""
        if flow is None:
            flow = self.get_flow(tenFirst, tenSecond, net)
        
        tenoutput = self.backwarp(tenSecond, flow)     
        tenMask = tenoutput[:, -1:, :, :]
        tenMask[tenMask > 0.999] = 1.0
        tenMask[tenMask < 1.0] = 0.0
        return tenoutput[:, :-1, :, :] * tenMask, tenMask

    def get_flow(self, tenFirst, tenSecond, net):
        """
        获取光流 - 使用 PyTorch PWC-Net
        输入输出都是 Jittor，内部调用 PyTorch
        """
        with jt.no_grad():
            # 注意：net 是 PyTorch 的 PWC-Net，已经在 eval 模式
            flow = self.estimate(tenFirst, tenSecond, net) 
        return flow