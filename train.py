#-*- encoding: UTF-8 -*-
import time
import sys, os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import numpy as np

os.environ['use_cuda'] = '1'           # 强制使用 CUDA
from util.util import calc_psnr as calc_psnr


def calc_ssim(sr, hr, ssim_criterion):
    """计算SSIM - 复用模型的SSIMLoss实例"""
    import jittor as jt
    
    with jt.no_grad():
        if not isinstance(sr, jt.Var):
            sr = jt.array(sr)
        if not isinstance(hr, jt.Var):
            hr = jt.array(hr)
        sr = sr / 255.0
        hr = hr / 255.0
        ssim_value = ssim_criterion(sr, hr) 
        return ssim_value.item()


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset_train = create_dataset(opt.dataset_name, 'train', opt)
    dataset_size_train = len(dataset_train)
    print('The number of training images = %d' % dataset_size_train)
    dataset_val = create_dataset(opt.dataset_name, 'val', opt)
    dataset_size_val = len(dataset_val)
    print('The number of val images = %d' % dataset_size_val)

    model = create_model(opt)
    model.setup(opt)
    
    
    visualizer = Visualizer(opt)
    
    # 计算总batch数
    total_batches_per_epoch = (dataset_size_train + opt.batch_size - 1) // opt.batch_size  # 向上取整
    total_iters = ((model.start_epoch * total_batches_per_epoch) // opt.print_freq) * opt.print_freq

    # 创建简要日志文件
    brief_log_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(brief_log_dir, exist_ok=True)
    brief_log_path = os.path.join(brief_log_dir, 'training_brief.txt')
    brief_log = open(brief_log_path, 'a')  # 追加模式
    brief_log.write(f'\n========== Training Started at {time.strftime("%Y-%m-%d %H:%M:%S")} ==========\n')
    brief_log.write(f'Total batches per epoch: {total_batches_per_epoch}\n')
    brief_log.flush()
    print(f'Brief training log will be saved to: {brief_log_path}')
    print(f'Total batches per epoch: {total_batches_per_epoch}')

    for epoch in range(model.start_epoch + 1, opt.niter + opt.niter_decay + 1):
        # training
        epoch_start_time = time.time()
        epoch_iter = 0
        model.train()

        iter_data_time = iter_start_time = time.time()
        for i, data in enumerate(dataset_train):
            if total_iters % opt.print_freq == 0:
                t_data = time.time() - iter_data_time
            
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)
            model.optimize_parameters()
   
            # 每10个batch打印简要信息到单独的txt文件
            if total_iters % 10 == 0:
                losses = model.get_current_losses()
                message = f'[Epoch {epoch}/{opt.niter + opt.niter_decay}][Batch {epoch_iter}/{total_batches_per_epoch}]  | L1: {losses["LiteISPNet_L1"]:.4f} | SSIM_Loss: {losses["LiteISPNet_SSIM"]:.4f} | VGG: {losses["LiteISPNet_VGG"]:.4f}\n'
                print(message.strip())
                brief_log.write(message)
                brief_log.flush()

            # 每print_freq个batch打印详细信息
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = time.time() - iter_start_time
                visualizer.print_current_losses(
                        epoch, epoch_iter, losses, t_comp, t_data, total_iters)
                if opt.save_imgs:
                    visualizer.display_current_results(
                        'train', model.get_current_visuals(), epoch)

            iter_data_time = time.time()
            
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d'
                  % (epoch, total_iters))
            model.save_networks(epoch)

        epoch_time = time.time() - epoch_start_time
        epoch_summary = f'End of epoch {epoch}/{opt.niter + opt.niter_decay} \t Time Taken: {epoch_time:.3f} sec\n'
        print(epoch_summary.strip())
        brief_log.write(epoch_summary)
        brief_log.flush()
        
        model.update_learning_rate()

        # val - 每个epoch结束后计算验证集指标
        if opt.calc_metrics:
            model.eval()
            val_iter_time = time.time()
            tqdm_val = tqdm(dataset_val, desc=f'Validating Epoch {epoch}')
            psnr_list = []
            ssim_list = []
            ssim_criterion = model.criterionSSIM
            for i, data in enumerate(tqdm_val):
                model.set_input(data)
                model.test()
                res = model.get_current_visuals()

                psnr_val = calc_psnr(res['data_dslr'], res['data_out'])
                psnr_list.append(psnr_val)

                ssim_val = calc_ssim(res['data_dslr'], res['data_out'],ssim_criterion)
                ssim_list.append(ssim_val)
                
            time_val = time.time() - val_iter_time
            mean_psnr = np.mean(psnr_list)
            mean_ssim = np.mean(ssim_list)

            visualizer.print_psnr(epoch, opt.niter + opt.niter_decay, time_val, mean_psnr)
            
            val_summary = f'[Validation] Epoch {epoch}/{opt.niter + opt.niter_decay} \t Time: {time_val:.3f}s \t PSNR: {mean_psnr:.4f} \t SSIM: {mean_ssim:.4f}\n'
            print(val_summary.strip())
            brief_log.write(val_summary)
            brief_log.flush()

        sys.stdout.flush()

    brief_log.write(f'\n========== Training Finished at {time.strftime("%Y-%m-%d %H:%M:%S")} ==========\n')
    brief_log.close()
    print(f'Training complete! Brief log saved to: {brief_log_path}')