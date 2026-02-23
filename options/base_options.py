import argparse
import os
import re
from util import util
import jittor as jt
import models
import time

def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', '1')

inf = float('inf')

class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # data parameters
        parser.add_argument('--dataroot', type=str, default='')
        parser.add_argument('--dataset_name', type=str, default=['eth'], nargs='+')
        parser.add_argument('--max_dataset_size', type=int, default=inf)
        parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale.')
        parser.add_argument('--mode', default='RGB', choices=['RGB', 'L', 'Y'],
                help='Currently, only RGB mode is supported.')
        parser.add_argument('--imlib', default='cv2', choices=['cv2', 'pillow'],
                help='Keep using cv2 unless encountered with problems.')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--patch_size', type=int, default=224)
        parser.add_argument('--shuffle', type=str2bool, default=True)
        parser.add_argument('-j', '--num_dataloader', default=4, type=int)
        parser.add_argument('--drop_last', type=str2bool, default=True)

        parser.add_argument('--gpu_ids', type=str, default='all',
                help='GPU selection for info display only. Jittor manages GPU automatically. '
                     'Use `--gpu_ids -1` to force CPU mode.')
        parser.add_argument('--checkpoints_dir', type=str, default='./ckpt')
        parser.add_argument('-v', '--verbose', type=str2bool, default=True)
        parser.add_argument('--suffix', default='', type=str)

        # model parameters
        parser.add_argument('--name', type=str, required=True,
                help='Name of the folder to save models and logs.')
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--load_path', type=str, default='',
                help='Will load pre-trained model if load_path is set')
        parser.add_argument('--load_iter', type=int, default=[0], nargs='+',
                help='Load parameters if > 0 and load_path is not set. '
                     'Set the value of `last_epoch`')
        parser.add_argument('--gcm_coord', type=str2bool, default=True)
        parser.add_argument('--pre_ispnet_coord', type=str2bool, default=True)
        parser.add_argument('--chop', type=str2bool, default=False)

        # training parameters
        parser.add_argument('--init_type', type=str, default='default',
                choices=['default', 'normal', 'xavier',
                         'kaiming', 'orthogonal', 'uniform'],
                help='`default` means using Jittor default init functions.')
        parser.add_argument('--init_gain', type=float, default=0.02)
        parser.add_argument('--optimizer', type=str, default='Adam',
                choices=['Adam', 'SGD', 'RMSprop'])
        parser.add_argument('--niter', type=int, default=1000)
        parser.add_argument('--niter_decay', type=int, default=0)
        parser.add_argument('--lr_policy', type=str, default='step')
        parser.add_argument('--lr_decay_iters', type=int, default=200)
        parser.add_argument('--lr', type=float, default=0.0001)

        # Optimizer
        parser.add_argument('--load_optimizers', type=str2bool, default=False,
                help='Loading optimizer parameters for continuing training.')
        parser.add_argument('--weight_decay', type=float, default=0)
        # Adam
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)
        # SGD & RMSprop
        parser.add_argument('--momentum', type=float, default=0)
        # RMSprop
        parser.add_argument('--alpha', type=float, default=0.99)

        # visualization parameters
        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--test_every', type=int, default=1000)
        parser.add_argument('--save_epoch_freq', type=int, default=1)
        parser.add_argument('--calc_metrics', type=str2bool, default=False)
        parser.add_argument('--save_imgs', type=str2bool, default=False)
        parser.add_argument('--visual_full_imgs', type=str2bool, default=False)

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=
                         argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options"""
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt'
                % ('train' if self.isTrain else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        opt.serial_batches = not opt.shuffle

        if self.isTrain and (opt.load_iter != [0] or opt.load_path != '') \
                and not opt.load_optimizers:
            util.prompt('You are loading a checkpoint and continuing training, '
                        'and no optimizer parameters are loaded. Please make '
                        'sure that the hyper parameters are correctly set.', 80)
            time.sleep(3)

        if opt.mode == 'RGB':
            opt.input_nc = opt.output_nc = 3
        else:
            opt.input_nc = opt.output_nc = 1
        opt.model = opt.model.lower()
        opt.name = opt.name.lower()

        scale_patch = {2: 96, 3: 144, 4: 192}
        if opt.patch_size is None:
            opt.patch_size = scale_patch[opt.scale]

        if opt.name.startswith(opt.checkpoints_dir):
            opt.name = opt.name.replace(opt.checkpoints_dir+'/', '')
            if opt.name.endswith('/'):
                opt.name = opt.name[:-1]

        if len(opt.dataset_name) == 1:
            opt.dataset_name = opt.dataset_name[0]
    
        if len(opt.load_iter) == 1:
            opt.load_iter = opt.load_iter[0]

        # process opt.suffix
        if opt.suffix != '':
            suffix = ('_' + opt.suffix.format(**vars(opt)))
            opt.name = opt.name + suffix

        self.print_options(opt)


        if opt.gpu_ids == '-1':
            # 强制 CPU 模式
            jt.flags.use_cuda = 0
            opt.gpu_ids = []
            util.prompt('You are using CPU mode (forced)')
        else:
            if jt.has_cuda:
                jt.flags.use_cuda = 1

                if opt.gpu_ids == 'all':
                    try:
                        import torch
                        gpu_count = torch.cuda.device_count()
                        opt.gpu_ids = list(range(gpu_count))
                    except:
                        # 如果无法获取，默认使用 GPU 0
                        opt.gpu_ids = [0]
                else:
                    p = re.compile('[^-0-9]+')
                    opt.gpu_ids = [int(i) for i in re.split(p, opt.gpu_ids) if int(i) >= 0]
                
                print('Jittor GPU mode enabled.')
                print(f'Using GPUs: {opt.gpu_ids}')
                print('Note: Jittor automatically manages GPU memory and scheduling.')
            else:
                jt.flags.use_cuda = 0
                opt.gpu_ids = []
                util.prompt('CUDA not available, using CPU mode')

        self.opt = opt
        return self.opt