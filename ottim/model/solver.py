import torch
import os
from munch import Munch
from utils.basic import *
import scipy.io

import numpy as np
from utils.eval import *
from utils.ssim import *
from utils.local_io import save_nii
from utils.interpolation import interpolation
from tqdm import tqdm
from model.loss import *
from termcolor import colored
import logging
import datetime
import nibabel as nib
from PIL import Image
from vedo import Volume, show
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from utils.early_stopping import EarlyStopping
#from model.cyclegan import CycleGANTranslator


class Solver:
    def __init__(self, dataset, model, MPR, args=None):
        self.dataset = dataset
        self.model = model
        self.MPR = MPR
        self.model_name = f"{model.name}_MPR" if MPR else model.name
        self.args = args
        self._initial_save_space()
        self.best_score = 0.0
        self.ssim_funct = SSIM(device=self.args.device)
        #self.cyclegan = CycleGANTranslator(...)

        if 'cuda' in self.args.device:
            self.model.to(self.args.device)

        # Inizializza gli handler prima dell'uso
        self.fh_train = None
        self.fh_test = None
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Configurazione logger centralizzata
        self.logger = logging.getLogger(self.model_name)
        self._init_logger()  # Chiamata unica alla configurazione del logger
        
        # Inizializzazione componenti per il training
        self.scheduler_g = None
        self.scheduler_d = None

        if args.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=2,  #  esempio: 2 valutazioni senza miglioramento
                min_delta=0.01,
                verbose=True
            )
        else:
            self.early_stopping = None

        self.lr_values_g = []
        self.lr_values_d = []

    def _init_logger(self):
        """Configurazione centralizzata del logger"""
        self.logger.setLevel(logging.INFO)
        
        # Rimuovi handler esistenti per evitare duplicati
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Handler per console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

        # Disabilita propagazione al root logger
        self.logger.propagate = False

    def _setup_file_logger(self, log_file_name, handler_type='train'):
        """Gestione file handler senza duplicati"""
        log_path = os.path.join(self.log_dir, log_file_name)
        
        # Rimuovi handler esistenti dello stesso tipo
        if handler_type == 'train' and self.fh_train:
            self.logger.removeHandler(self.fh_train)
        elif handler_type == 'test' and self.fh_test:
            self.logger.removeHandler(self.fh_test)

        # Crea nuovo handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(self.formatter)

        # Aggiorna riferimenti
        if handler_type == 'train':
            self.fh_train = fh
        else:
            self.fh_test = fh

        self.logger.addHandler(fh)
        self.logger.info(f"Logging to file: {log_path}")

    def _adjust_learning_rate(self, optimizer, epoch_id, raw_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
        if epoch_id <= 100:
            lr = raw_lr * (0.1 ** (epoch_id // 50))
        else:
            lr = raw_lr * (0.1 ** 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
        
    def train(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_file_logger(f'training_{timestamp}.log', 'train')
        sampler = self.dataset.train_sampler

        # Determina quale modalità di learning rate utilizzare
        self.use_manual_lr = self.args.lr_scheduler == 'manual'
        self.use_plateau_lr = self.args.lr_scheduler == 'plateau'
        self.use_cosine_lr = self.args.lr_scheduler == 'cosine'
        
        if not (self.use_manual_lr or self.use_plateau_lr or self.use_cosine_lr):
            self.logger.warning(f"Modalità LR scheduler '{self.args.lr_scheduler}' non riconosciuta, usando 'manual'")
            self.use_manual_lr = True
            self.use_plateau_lr = False
            self.use_cosine_lr = False

        # Log della modalità di LR scelta
        self.logger.info(f"Utilizzando modalità learning rate: {self.args.lr_scheduler}")

        # Inizializza le liste per il tracking
        train_g_loss = []
        train_d_loss = []
        val_scores = []
        val_epochs = []
        val_psnr = []
        val_ssim = []
        val_dice = []
        self.lr_values_g = [] 
        self.lr_values_d = []

        self.logger.info("Starting training with the following arguments:")
        for arg, value in vars(self.args).items():
            self.logger.info(f"{arg}: {value}")

        # Prima valutazione iniziale
        eval_scores = self.eval(sampler=self.dataset.val_sampler, get_eval_score=True, save_generations=False)
        self.print_eval_scores(eval_scores, title='Val_0')
        self.overall_score = eval_scores['overall_score']

        optim_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.args.g_lr)
        optim_d = torch.optim.SGD(self.model.discriminator.parameters(), lr=self.args.d_lr)
        
        # Inizializza gli scheduler in base alla modalità scelta
        if self.use_plateau_lr:
            self.scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_g, 
                mode='max',       
                factor=0.5,       
                patience=2,      
                threshold=0.01,
                min_lr=1e-6
            )
            self.scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_d,
                mode='max',
                factor=0.5,
                patience=2,
                threshold=0.01,
                min_lr=1e-6
            )
        elif self.use_cosine_lr:
            # Nuovo scheduler Cosine Annealing con Warm Restarts
            self.scheduler_g = CosineAnnealingWarmRestarts(
                optim_g,
                T_0=50,      # Durata del primo ciclo (50 epoche)
                T_mult=2,    # Ogni ciclo successivo è 2 volte più lungo
                eta_min=1e-6 # LR minimo
            )
            self.scheduler_d = CosineAnnealingWarmRestarts(
                optim_d,
                T_0=50,
                T_mult=2,
                eta_min=1e-6
            )
        
        # Resume logic
        start_epoch = 1
        if self.args.resume:
            start_epoch = self._load_ckpt(self.args.resume)
            self.logger.info(f"Ripresa training da epoca {start_epoch}")
            start_epoch += 1  

        for epoch_id in range(start_epoch, self.args.train_n + 1):
            # Aggiorna learning rate in base alla modalità
            if self.use_manual_lr:
                optim_g = self._adjust_learning_rate(optim_g, epoch_id, self.args.g_lr)
                optim_d = self._adjust_learning_rate(optim_d, epoch_id, self.args.d_lr)
            elif self.use_cosine_lr:
                # Per Cosine Annealing, aggiorniamo all'inizio di ogni epoca
                self.scheduler_g.step()
                self.scheduler_d.step()
            
            # Salva il learning rate corrente per il plot
            self.lr_values_g.append(optim_g.param_groups[0]['lr'])
            self.lr_values_d.append(optim_d.param_groups[0]['lr'])

            with tqdm(total=sampler.data_n) as epoch_pbar:
                d_loss_list = []
                g_loss_list = []
                for batch_id in range(sampler.batch_n):
                    batch = sampler.get_batch()
                    px_tensor = torch.tensor(np.array(batch['Ideal_PX']), dtype=torch.float,
                                            device=self.args.device)
                    
                    gt_tensor = torch.tensor(np.array(batch['MPR']), dtype=torch.float, device=self.args.device) if self.MPR \
                        else torch.tensor(np.array(batch['CBCT']), dtype=torch.float, device=self.args.device)

                    generations = self.model.generate(px_tensor, VAL=False)
                    loss_g = proj_loss(generations, gt_tensor) + rec_loss(generations, gt_tensor)
                    g_loss_list.append(loss_g.item())

                    if epoch_id > self.args.d_start_n:
                        loss_g += self.model.discriminator.inference(generations) * 0.05

                    optim_g.zero_grad()
                    loss_g.backward()
                    optim_g.step()

                    # Update discriminator
                    if epoch_id > self.args.d_start_n:
                        loss_d = self.model.discriminator.discriminate(generations.detach(), gt_tensor)
                        optim_d.zero_grad()
                        loss_d.backward()
                        optim_d.step()
                        loss_d = loss_d.item()
                    else:
                        loss_d = 0.0
                    d_loss_list.append(loss_d)

                    # Update progress bar
                    desc = f'Epoch:{epoch_id:03d}|loss_g {loss_g:.4f}, loss_d {loss_d:.4f}'
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(px_tensor.shape[0])
                
                # Registra le loss medie
                epoch_g_loss = np.mean(g_loss_list)
                epoch_d_loss = np.mean(d_loss_list)
                train_g_loss.append(epoch_g_loss)
                train_d_loss.append(epoch_d_loss)

                epoch_pbar.set_description(f'Epoch:{epoch_id:03d}|loss_g {epoch_g_loss:.4f}, loss_d {epoch_d_loss:.4f}')
                epoch_pbar.close()

                # Log epoca e LR
                self.logger.info(f"Epoch:{epoch_id:03d}|G_loss: {epoch_g_loss:.4f}, D_loss: {epoch_d_loss:.4f}")
                self.logger.info(f"Current LR Generator: {optim_g.param_groups[0]['lr']:.2e}")
                self.logger.info(f"Current LR Discriminator: {optim_d.param_groups[0]['lr']:.2e}")

            # Validazione periodica
            if epoch_id % self.args.val_n == 0:
                eval_scores = self.eval(
                    sampler=self.dataset.val_sampler, 
                    get_eval_score=True,
                    save_generations=False, 
                    save_dir=f'Val_{epoch_id:03d}'
                )
                self.print_eval_scores(eval_scores, title=f'Val_{epoch_id:03d}')
                
                # Aggiorna gli scheduler solo per ReduceLROnPlateau
                if self.use_plateau_lr and self.scheduler_g is not None:
                    self.scheduler_g.step(eval_scores['overall_score'])
                    self.scheduler_d.step(eval_scores['overall_score'])
                
                # Salva metriche
                val_scores.append(eval_scores['overall_score'])
                val_psnr.append(eval_scores['psnr_mean'])
                val_ssim.append(eval_scores['ssim_mean'])
                val_dice.append(eval_scores['dice_mean'])
                val_epochs.append(epoch_id)


    # Early stopping e best model SOLO dopo che il discriminatore è attivo
                if self.early_stopping and epoch_id >= self.args.d_start_n + self.args.val_n:
                    is_best = self.early_stopping(eval_scores['overall_score'], epoch_id, self.logger)
                    if is_best:
                        self._save_ckpt(0, optim_g, optim_d)
                    
                    if self.early_stopping.early_stop:
                        self.logger.info(f"Early stopping attivato all'epoca {epoch_id}")
                        break

            # Salvataggio checkpoint
            if epoch_id % self.args.save_n == 0:
                self._save_ckpt(epoch_id, optim_g, optim_d)
        
        
        # Caricamento finale del miglior modello
        if self.early_stopping:
            self.logger.info("Caricamento miglior modello salvato")
            self._load_ckpt(0)

        # Plot finale con suffisso basato sul tipo di scheduler
        suffix = self.args.lr_scheduler
        self._plot_training_curves(
            train_g_loss, 
            train_d_loss, 
            val_scores,
            val_psnr,
            val_ssim,
            val_dice,
            val_epochs,
            suffix
        )
        
        self.logger.info("Training finished.")

    def _plot_training_curves(self, train_g, train_d, val_scores, val_psnr, val_ssim, val_dice, val_epochs, suffix):
        plt.figure(figsize=(18, 12))
        
        # Subplot 1: Loss di training
        plt.subplot(2, 2, 1)
        plt.plot(train_g, 'b', label='Generator Loss')
        
        # Plot discriminator loss solo da d_start_n in poi
        d_start_n = self.args.d_start_n
        epochs = np.arange(len(train_g))
        plt.plot(epochs[d_start_n:], train_d[d_start_n:], 'r', label='Discriminator Loss')
        plt.title('Training Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Metriche di validazione
        if val_epochs:
            plt.subplot(2, 2, 2)
            plt.plot(val_epochs, val_scores, 'g-o', label='Overall Score')
            plt.plot(val_epochs, val_psnr, 'b-s', label='PSNR')
            plt.plot(val_epochs, val_ssim, 'r-^', label='SSIM')
            plt.plot(val_epochs, val_dice, 'm-d', label='Dice')
            plt.title('Validation Performance')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
        
        # Subplot 3: Learning Rate (nuovo)
        plt.subplot(2, 2, 3)
        plt.plot(self.lr_values_g, 'g-', label='Generator LR')
        plt.plot(self.lr_values_d, 'r--', label='Discriminator LR')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('LR')
        plt.yscale('log')  # Scala logaritmica per il LR
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Salvataggio
        plot_path = os.path.join(self.log_dir, f'training_metrics_{suffix}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        self.logger.info(f"Saved training curves to: {plot_path}")

    # return evaluation result
    def eval(self, sampler, get_eval_score=True, save_generations=False, save_dir=None):
        psnr_list = []
        ssim_list = []
        dice_list = []
        
        with torch.no_grad(): # Disabilita il calcolo dei gradienti
            for batch_id in range(sampler.batch_n):
                # sample batch
                batch = sampler.get_batch()
                batch_img = batch['Ideal_PX']
                batch_size = len(batch_img)
                batch_shape = batch['PriorShape'] if self.MPR else None
                px_tensor = torch.tensor(np.array(batch_img), dtype=torch.float, device=self.args.device)
                generations = self.model.generate(px_tensor, VAL=True)
                generations_cpu = generations.detach() if 'cpu' in self.args.device else generations.detach().cpu().numpy()
                
                # if the shape is provided, registrant the image back to the old space
                if batch_shape:
                    generation_list = []
                    for generation_cpu, prior_shape in zip(generations_cpu, batch_shape):
                        # ignore part of the generation slices
                        ignore_slice_n = 15
                        generation_cpu[:ignore_slice_n, :, :] = -1
                        generation_cpu[-ignore_slice_n:, :, :] = -1
                        generation_list.append(interpolation(generation_cpu, prior_shape))
                    generations_cpu = generation_list

                for item_id in range(batch_size):
                    generation = generations_cpu[item_id]
                    if get_eval_score:
                        # evaluate performance
                        CBCT = batch['CBCT'][item_id]
                        MPR = batch['MPR'][item_id]
                        batch_shape = batch['PriorShape']
                        psnr_list.append(get_psnr(generation, CBCT))
                        ssim_list.append(self.ssim_funct.eval_ssim(generation, CBCT))
                        dice_list.append(
                            get_dice(generation > -0.8, interpolation(MPR, batch_shape[item_id]) > -0.5))
                    
                    # save results
                    if save_generations:
                        generation_dir = join_path(self.result_dir, save_dir)
                        os.makedirs(generation_dir, exist_ok=True)
                        case_id = batch['Case_ID'][item_id]
                        generation_img = np.array((generation + 1) * 2000, dtype=np.uint16)
                        #generation_img[generation_img < 0] = 0
                       # generation_img = normalize_case(generation_img)
                       # generation_img[generation_img < 500] = 0
                        #generation_img = clamp_case(generation_img, 4000)
                        generation_img = np.transpose(generation_img, (2, 0, 1))
                        save_nii(generation_img, np.eye(4), 'case_%03d.nii.gz' % case_id, generation_dir)

        if get_eval_score:
            psnr_mean, psnr_std = np.mean(psnr_list), np.std(psnr_list)
            ssim_mean, ssim_std = np.mean(ssim_list), np.std(ssim_list)
            dice_mean, dice_std = np.mean(dice_list), np.std(dice_list)
            overall_score = self.get_overall_score(psnr_mean, ssim_mean, dice_mean)
            
            # Log the evaluation scores
            self.logger.info(
                f"Eval - PSNR: {psnr_mean:.2f} +/- {psnr_std:.2f}, SSIM: {ssim_mean:.2f} +/- {ssim_std:.2f}, Dice: {dice_mean:.2f} +/- {dice_std:.2f}, Overall: {overall_score:.2f}")
            
            return Munch(psnr_mean=psnr_mean, psnr_std=psnr_std,
                        ssim_mean=ssim_mean, ssim_std=ssim_std,
                        dice_mean=dice_mean, dice_std=dice_std,
                        overall_score=overall_score)
        else:
            return None

    # test simulated case
    def test(self):
        self.logger.info("Starting testing.")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_file_logger(f'testing_{timestamp}.log', 'test')
        self.use_plateau_lr = False
        self.use_cosine_lr = False
        self._load_ckpt(epoch=0)
        eval_scores = self.eval(sampler=self.dataset.test_sampler, get_eval_score=False, save_generations=True,      # get_eval_score=  True se si vogliono le metriche
                            save_dir='TEST')

        # 2. Seleziona un file casuale dalla directory di output
        generation_dir = os.path.join(self.result_dir, 'TEST')
        nii_files = [os.path.join(generation_dir, f) for f in os.listdir(generation_dir)
                    if f.endswith('.nii.gz')]
        if nii_files:
            # 3. Visualizzazione interattiva con Vedo
            random_file = random.choice(nii_files)
            vol = Volume(random_file)
            show(vol, bg="white", axes=1, viewup="z", title="Visualizzazione Volume Generato")
        else:
            self.logger.warning("Nessun file NIfTI generato per la visualizzazione")

#        self.print_eval_scores(eval_scores, title='TEST')
        self.logger.info("Testing finished.")



    def infer(self, mat_path, output_dir=None):
        """
        Struttura simile a `test`, ma usa un singolo file .mat per l'inferenza.
        """
        self.logger.info("Starting structured MAT-based inference.")
        self._load_ckpt(epoch=0)
        self.use_plateau_lr = False
        self.use_cosine_lr = False

        mat = scipy.io.loadmat(mat_path)

        if 'Ideal_PX' in mat:
            px_array = mat['Ideal_PX']
        elif 'PX' in mat:
            px_array = mat['PX']
        else:
            raise KeyError("Il file MAT deve contenere 'Ideal_PX' o 'PX'")

        if 'AvgPriorShape' in mat:
            prior_shape = mat['AvgPriorShape']
        elif 'AvgSplineCtrlPoints' in mat:
            prior_shape = mat['AvgSplineCtrlPoints']
        else:
            raise KeyError("Il file MAT deve contenere 'AvgSplineCtrlPoints' o 'AvgPriorShape'")

        # Normalizza immagine
        px_array = px_array.astype(np.float32)
        if px_array.max() > 1.0:
            px_array /= 255.0

        # Invia immagine alla rete
        px_tensor = torch.tensor(px_array[np.newaxis, np.newaxis, ...],
                                dtype=torch.float32,
                                device=self.args.device)
        with torch.no_grad():
            generation = self.model.generate(px_tensor, VAL=True)
            generation_np = generation.squeeze().cpu().numpy()


        # Ricostruzione CBCT
        reconstructed = interpolation(generation_np, prior_shape)

        # Salva NIfTI
# Salva NIfTI
        if output_dir is None:
            output_dir = os.path.join(self.result_dir, "INFER")
        os.makedirs(output_dir, exist_ok=True)
        nii_array = np.array((reconstructed + 1) * 2000, dtype=np.uint16)
        nii_array = np.transpose(nii_array, (2, 0, 1))

        filename = "cbct_from_mat.nii.gz"
        save_nii(nii_array, np.eye(4), filename, output_dir)

        save_path = os.path.join(output_dir, filename)


        # Visualizzazione
        try:
            vol = Volume(save_path)
            show(vol, bg="white", axes=1, viewup="z", title="CBCT simulata da .mat")
        except Exception as e:
            self.logger.warning(f"Errore visualizzazione: {str(e)}")

        self.logger.info(f"Inferenza MAT completata. Risultato salvato in: {save_path}")

    #def infer(self, png_dir, curve_dir, output_dir):
    #    """Esegue inferenza da immagini PNG e curve dentali"""
    #    self.logger.info("Starting PNG-based inference.")
    #    self._load_ckpt(epoch=0)
    #    
    #    # Crea la directory di output
    #    os.makedirs(output_dir, exist_ok=True)
    #     
#
#
    #    # integrare la cyclegan translator
    #    # real_px = load_image(...)
    #    # sim_px = self.cyclegan.translate(real_px)  # Traduzione dominio
    #    # generation = self.model.generate(sim_px)
#
    #    # Processa tutte le immagini PNG
    #    for png_file in os.listdir(png_dir):
    #        if png_file.lower().endswith(('.png', '.jpg', '.jpeg')):
    #            # Carica e preprocessa l'immagine
    #            img_path = os.path.join(png_dir, png_file)
    #            img = Image.open(img_path).convert('L')  # Converti in scala di grigi
    #            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalizza [0,1]
    #            
    #            # Carica la curva corrispondente
    #            base_name = os.path.splitext(png_file)[0]
    #            curve_path = os.path.join(curve_dir, f"{base_name}_curve.txt")
    #            prior_shape = np.loadtxt(curve_path)
    #            
    #            # Converti in tensore
    #            px_tensor = torch.tensor(img_array[np.newaxis, np.newaxis, ...],
    #                                    dtype=torch.float32,
    #                                    device=self.args.device)
    #            
    #            # Genera la ricostruzione
    #            with torch.no_grad():
    #                generation = self.model.generate(px_tensor, VAL=True)
    #                generation_np = generation.squeeze().cpu().numpy()
    #            
    #            # Applica la curva
    #            ignore_slice_n = 15
    #            generation_np[:ignore_slice_n, :, :] = -1
    #            generation_np[-ignore_slice_n:, :, :] = -1
    #            reconstructed = interpolation(generation_np, prior_shape)
    #            
    #            # Converti e salva il risultato
    #            output_path = os.path.join(output_dir, f"{base_name}_cbct.nii.gz")
    #            generation_img = np.array((reconstructed + 1) * 2000, dtype=np.uint16)
    #            generation_img = np.transpose(generation_img, (2, 0, 1))
    #            generation_dir = os.path.join(self.result_dir, 'INFER')
    #            save_nii(generation_img, np.eye(4), 'case.nii.gz', generation_dir)
#
#
    #    
    #    nii_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
    #    if nii_files:
    #        try:
    #            # Selezione e visualizzazione randomica
    #            selected_file = os.path.join(output_dir, random.choice(nii_files))
    #            self.logger.info(f"Visualizzazione: {selected_file}")
    #            vol = Volume(selected_file)
    #            show(vol,
    #                bg="white",
    #                axes=1,
    #                viewup="z",
    #                interactive=True  # Blocca l'esecuzione finché la finestra è aperta
    #            )
    #        except Exception as e:
    #            self.logger.error(f"Errore visualizzazione: {str(e)}")
    #    else:
    #        self.logger.warning("Nessun file generato trovato per la visualizzazione.")
    #        
    #    self.logger.info("Inference completata. Risultati salvati in: %s", output_dir)
#
    def _save_ckpt(self, epoch=0, optim_g=None, optim_d=None):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': optim_g.state_dict() if optim_g else None,
            'optimizer_d_state_dict': optim_d.state_dict() if optim_d else None,
            'scheduler_g_state_dict': self.scheduler_g.state_dict() if self.scheduler_g else None,
            'scheduler_d_state_dict': self.scheduler_d.state_dict() if self.scheduler_d else None,
            'best_score': self.best_score,
            'lr_scheduler_type': self.args.lr_scheduler  # Salva anche il tipo di scheduler
        }
        ckpt_name = 'model_best.pth.tar' if epoch == 0 else f'model_{epoch}.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        torch.save(state, ckpt_path)

    def _load_ckpt(self, epoch=0):
        ckpt_name = 'model_best.pth.tar' if epoch == 0 else f'model_{epoch}.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} non trovato")
        
        state = torch.load(ckpt_path, map_location=self.args.device)
        self.model.load_state_dict(state['model_state_dict'])
        
        # Carica lo stato degli scheduler solo se necessario
#        if self.use_plateau_lr or self.use_cosine_lr:
#            if 'scheduler_g_state_dict' in state and self.scheduler_g:
#                self.scheduler_g.load_state_dict(state['scheduler_g_state_dict'])
#            if 'scheduler_d_state_dict' in state and self.scheduler_d:
#                self.scheduler_d.load_state_dict(state['scheduler_d_state_dict'])
                
        return epoch

    @staticmethod
    def print_eval_scores(eval_scores, title):
        psnr_mean, psnr_std = eval_scores['psnr_mean'], eval_scores['psnr_std']
        ssim_mean, ssim_std = eval_scores['ssim_mean'], eval_scores['ssim_std']
        dice_mean, dice_std = eval_scores['dice_mean'], eval_scores['dice_std']
        overall_score = eval_scores['overall_score']
        desc = f'{title}|PSNR: {psnr_mean:.2f} {psnr_std:.2f}, SSIM: {ssim_mean:.2f} {ssim_std:.2f}, Dice: {dice_mean:.2f} {dice_std:.2f}, AVG: {overall_score:.2f}'
        print(colored(desc, 'blue'))

    @staticmethod
    def get_overall_score(psnr, ssim, dice):
        return (psnr / 20 + ssim/100 + dice/100) / 3 * 100
       # return ( 0.2 * (psnr / 20) + 0.2 * (ssim/100) + 0.6 * (dice/100) ) * 100

    def _initial_save_space(self):
        save_space = check_dir(join_path('output', self.model_name))
        self.log_dir = check_dir(join_path(save_space, 'log'))
        self.ckpt_dir = check_dir(join_path(save_space, 'ckpt'))
        self.result_dir = check_dir(join_path(save_space, 'result'))
