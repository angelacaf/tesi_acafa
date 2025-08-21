#python test.py --dataroot ./datasets/px/testA --name px_cyclegan --model test --no_dropout --model_suffix _A
# 
import os
import numpy as np
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from skimage.metrics import structural_similarity as ssim

def get_psnr(img_1, img_2, SCALE=255.0):  # SCALE corretto per immagini 8-bit
    mse = np.mean((img_1.astype(float) - img_2.astype(float))**2)
    return 10 * np.log10((SCALE**2) / (mse + 1e-8))  # Formula corretta

def get_dice(gt_img, pr_img, threshold=127):
    mask_1 = (gt_img > threshold).astype(np.uint8)
    mask_2 = (pr_img > threshold).astype(np.uint8)
    intersection = np.sum(mask_1 & mask_2)
    return (2.0 * intersection) / (np.sum(mask_1) + np.sum(mask_2) + 1e-8) * 100

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def tensor2im(input_image, imtype=np.uint8):
    """Converte un tensor PyTorch in un numpy array per il calcolo delle metriche"""
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)


    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt)
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    if opt.load_iter > 0:
        web_dir = f'{web_dir}_iter{opt.load_iter}'
    
    print('Creating web directory:', web_dir)
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')

    psnr_values = []
    ssim_values = []
    dice_values = []

    metrics_file = os.path.join(web_dir, 'metrics.txt')
    with open(metrics_file, 'w') as metrics_out:
        metrics_out.write("Image\tPSNR\tSSIM\tDice\n")

        if opt.eval:
            model.eval()

        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break

            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()

            real_key = 'real_A' if 'real_A' in visuals else 'real'
            fake_key = 'fake_B' if 'fake_B' in visuals else 'fake'

            if real_key in visuals and fake_key in visuals:
                real_img = tensor2im(visuals[real_key])
                fake_img = tensor2im(visuals[fake_key])

                try:
                    # Calcola PSNR con formula corretta
                    psnr_value = get_psnr(real_img, fake_img, SCALE=255.0)
                    
                    # Calcola SSIM con la tua classe
                    if len(real_img.shape) == 3 and real_img.shape[2] == 3:
                        real_img_gray = np.mean(real_img, axis=2).astype(np.uint8)
                        fake_img_gray = np.mean(fake_img, axis=2).astype(np.uint8)
                        ssim_value = ssim(real_img_gray, fake_img_gray, data_range=255)
                    else:
                        ssim_value = ssim(real_img, fake_img, data_range=255)
                        
                    # Calcola Dice con soglia corretta
                    dice_value = get_dice(real_img, fake_img, threshold=127)

                    psnr_values.append(psnr_value)
                    ssim_values.append(ssim_value)
                    dice_values.append(dice_value)

                    metrics_out.write(f"{img_path[0]}\t{psnr_value:.4f}\t{ssim_value:.4f}\t{dice_value:.4f}\n")
                    
                    if i % 1 == 0:
                        print(f'Processed {i:04d} images | PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.2f}, Dice: {dice_value:.2f}%')

                except Exception as e:
                    print(f"Error processing image {img_path[0]}: {str(e)}")

            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

        # Calcola e salva le metriche medie
        avg_psnr = np.nanmean(psnr_values) if psnr_values else 0
        avg_ssim = np.nanmean(ssim_values) if ssim_values else 0
        avg_dice = np.nanmean(dice_values) if dice_values else 0

        metrics_out.write(f"\nAverage\t{avg_psnr:.4f}\t{avg_ssim:.4f}\t{avg_dice:.4f}")
        print(f'\nFinal metrics | PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.2f}, Dice: {avg_dice:.2f}%')

    webpage.save()
