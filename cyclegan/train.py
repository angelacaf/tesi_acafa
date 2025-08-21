import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure



def get_psnr(img_1, img_2, SCALE=255.0):
    mse = np.mean((img_1.astype(float) - img_2.astype(float))**2)
    return 10 * np.log10((SCALE**2) / (mse + 1e-8)) if mse != 0 else float('inf')

def get_dice(gt_img, pr_img, threshold=127):
    mask_1 = (gt_img > threshold).astype(np.uint8)
    mask_2 = (pr_img > threshold).astype(np.uint8)
    intersection = np.sum(mask_1 & mask_2)
    return (2.0 * intersection) / (np.sum(mask_1) + np.sum(mask_2) + 1e-8) * 100

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    # Inizializza le metriche
    device = 'cuda' if opt.gpu_ids else 'cpu'
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_values = []
    ssim_values = []
    dice_values = []
    epochs_list = []

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        for i, data in enumerate(dataset):
            # Loop principale invariato
            model.set_input(data)
            model.optimize_parameters()


            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()



            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            
            
            
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % 20== 0:
            with torch.no_grad():
                # Usa l'ultimo batch processato
                real_A = model.real_A.detach()
                fake_B = model.fake_B.detach()
                
                p = get_psnr(fake_B, real_A)
                s = ssim(fake_B, real_A)
                d = get_dice(fake_B, real_A)
                psnr_values.append(p.item())
                ssim_values.append(s.item())
                dice_values.append(d.item())
                epochs_list.append(epoch)
                # Crea e salva il plot
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(epochs_list, psnr_values, 'b-o', label='PSNR')
                plt.plot(epochs_list, ssim_values, 'r-s', label='SSIM')
                plt.xlabel('Epoca')
                plt.ylabel('Valore')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(epochs_list, dice_values, 'g-D', label='Dice Score')
                plt.xlabel('Epoca')
                plt.ylabel('Valore')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'./checkpoints/{opt.name}/metrics_epoch_{epoch}.png')
                plt.close()
        model.update_learning_rate()    # L'aggiornamento del learning rate dovrebbe avvenire alla fine dell'epoca, dopo aver completato il training su tutti i dati. Se aggiornato all'inizio, potrebbe causare instabilità nel training.
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    visualizer.save_final_loss_plot()

#           python -m visdom.server` and click the URL http://localhost:8097.  altrimenti
#           python train.py --dataroot ./datasets/px --name px_cyclegan --model cycle_gan --display_id 0
# 
# python train.py --dataroot ./datasets/px --name px_cyclegan --display_id 1 

