import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import kornia
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim

def tensor2im(input_image, imtype=np.uint8):
    """Converte un tensor PyTorch in un numpy array formattato correttamente"""
    if isinstance(input_image, torch.Tensor):
        # Gestione esplicita delle dimensioni
        image_tensor = input_image.detach().cpu()
        
        if image_tensor.dim() == 4:  # Formato (B,C,H,W)
            image_tensor = image_tensor.squeeze(0)
        
        # Converti in numpy e gestisci canali
        image_numpy = image_tensor.numpy()
        
        # Gestione immagini monocromatiche
        if image_numpy.shape[0] == 1:  # Immagine in scala di grigi
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            
        # Trasposizione assi (C,H,W) -> (H,W,C) se necessario
        if image_numpy.shape[0] == 3:
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        
        # Normalizzazione finale [-1,1] -> [0,255]
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    
    return image_numpy.astype(imtype)

def get_psnr(img_1, img_2, SCALE=255.0):
    mse = np.mean((img_1.astype(float) - img_2.astype(float))**2)
    return 10 * np.log10((SCALE**2) / (mse + 1e-8)) if mse != 0 else float('inf')

def get_dice(gt_img, pr_img, threshold=127):
    mask_1 = (gt_img > threshold).astype(np.uint8)
    mask_2 = (pr_img > threshold).astype(np.uint8)
    intersection = np.sum(mask_1 & mask_2)
    return (2.0 * intersection) / (np.sum(mask_1) + np.sum(mask_2) + 1e-8) * 100

def normalized_cross_correlation(x, y):
    """Calcola la NCC tra due immagini normalizzate in [-1, 1]"""
    x = x.cpu()
    y = y.cpu()
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    numerator = torch.sum((x - mean_x) * (y - mean_y))
    denominator = torch.sqrt(torch.sum((x - mean_x)**2) * torch.sum((y - mean_y)**2))
    return numerator / (denominator + 1e-8)

def regularizzazione(flow):
    """Calcola la regolarizzazione del campo di flusso"""
    dx = flow[:, :, :, :-1] - flow[:, :, :, 1:]
    dy = flow[:, :, :-1, :] - flow[:, :, 1:, :]
    return torch.mean(dx**2) + torch.mean(dy**2)

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, fixed, registered, flow):
        ncc = -normalized_cross_correlation(fixed, registered)
        mse = F.mse_loss(fixed, registered)
        reg = regularizzazione(flow)
        total = self.alpha*ncc + self.beta*mse + self.gamma*reg
        return total, ncc, mse, reg


def calculate_ncc(img1, img2):
    """Versione ottimizzata con gestione automatica del dispositivo"""
    img1 = img1.cpu()
    img2 = img2.cpu()
    mean1 = img1.mean()
    mean2 = img2.mean()
    numerator = ((img1 - mean1) * (img2 - mean2)).sum()
    denominator = torch.sqrt(((img1 - mean1)**2).sum() * ((img2 - mean2)**2).sum())
    return (numerator / (denominator + 1e-8)).item()

def calculate_overlap_metrics(fixed, registered):
    # Converti i tensori PyTorch in numpy array uint8
    fixed_np = tensor2im(fixed)
    registered_np = tensor2im(registered)
    
    # Calcola SSIM
    ssim_value = ssim(fixed_np, registered_np, 
                     data_range=255, 
                     channel_axis=-1 if fixed_np.shape[-1] == 3 else None)
    
    # Calcola le altre metriche
    return {
        'SSIM': ssim_value,
        'PSNR': get_psnr(fixed_np, registered_np),
        'Dice': get_dice(fixed_np, registered_np),
        'NCC': calculate_ncc(fixed, registered)
    }

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Genera descrittori di feature efficienti applicando max-pooling e avg-pooling 
        # sull'asse dei canali
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatena i risultati
        attn = torch.cat([max_pool, avg_pool], dim=1)
        
        # Elabora con una convoluzione e applica sigmoid
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        
        # Moltiplica il tensore originale per la mappa di attenzione
        return x * attn


class AttentionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_attention = use_attention
        if use_attention:
            self.spatial_attn = SpatialAttention(kernel_size=7)
            
    def forward(self, x):
        x = self.conv(x)
        if self.use_attention:
            x = self.spatial_attn(x)
        return x


#class UNet(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.enc1 = AttentionConvBlock(2, 64, use_attention=True)
#        self.pool1 = nn.MaxPool2d(2)
#        self.enc2 = AttentionConvBlock(64, 128, use_attention=True)
       # resto del codice invariato

       
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(2, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)


        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.flow = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.tanh(self.flow(d1)) * 0.05

class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, moving, flow):
        B, C, H, W = moving.size()
        grid = kornia.utils.create_meshgrid(H, W, device=moving.device).repeat(B,1,1,1)
        new_grid = grid + flow.permute(0,2,3,1)
        return F.grid_sample(moving, new_grid, align_corners=True, mode='bicubic')

class RegistrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
        self.transformer = SpatialTransformer()

    def forward(self, fixed, moving):
        x = torch.cat([fixed, moving], dim=1)
        flow = self.unet(x)
        return self.transformer(moving, flow), flow

class DentalDataset(Dataset):
    def __init__(self, fixed_dir, moving_dir):
        self.fixed_dir = fixed_dir
        self.moving_dir = moving_dir
        self.files = sorted([f for f in os.listdir(fixed_dir) if f.endswith('.png')])
        for f in self.files:
            if not os.path.exists(os.path.join(moving_dir, f)):
                raise FileNotFoundError(f"File corrispondente mancante: {f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fixed = Image.open(os.path.join(self.fixed_dir, self.files[idx])).convert('L')
        moving = Image.open(os.path.join(self.moving_dir, self.files[idx])).convert('L')
        fixed_tensor = 2 * (torch.from_numpy(np.array(fixed, dtype=np.float32)/255) - 0.5)
        moving_tensor = 2 * (torch.from_numpy(np.array(moving, dtype=np.float32)/255) - 0.5)
        return fixed_tensor.unsqueeze(0), moving_tensor.unsqueeze(0)

def visualize_results(fixed, moving, registered, metrics_before, metrics_after):
    plt.figure(figsize=(18,6))
    images = [
        ('Fixed', fixed.squeeze()),
        ('Moving', moving.squeeze()),
        ('Registered', registered.squeeze()),
        ('Difference', np.abs(fixed.squeeze() - registered.squeeze()))
    ]
    for i, (title, img) in enumerate(images):
        plt.subplot(1,4,i+1)
        plt.imshow(img, cmap='gray' if i <3 else 'hot')
        plt.title(title)
        plt.axis('off')
        if i == 3:
            plt.colorbar()
    plt.suptitle(
        f"SSIM: {metrics_before['SSIM']:.3f}→{metrics_after['SSIM']:.3f} | "
        f"PSNR: {metrics_before['PSNR']:.2f}→{metrics_after['PSNR']:.2f} | "
        f"Dice: {metrics_before['Dice']:.3f}→{metrics_after['Dice']:.3f} | "
        f"NCC: {metrics_before['NCC']:.3f}→{metrics_after['NCC']:.3f}",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_loss_components(total_losses, ncc_losses, mse_losses, reg_losses):
    epochs = range(1, len(total_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_losses, 'k-', label='Loss Totale')
    plt.plot(epochs, ncc_losses, 'r-', label='NCC')
    plt.plot(epochs, mse_losses, 'g-', label='MSE')
    plt.plot(epochs, reg_losses, 'b-', label='Regolarizzazione')
    plt.title('Andamento delle componenti della loss durante il training')
    plt.xlabel('Epoche')
    plt.ylabel('Valore Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo utilizzato: {device}")
    
    fixed_dir = './fixed'
    moving_dir = './moving'
    if not os.path.isdir(fixed_dir) or not os.path.isdir(moving_dir):
        raise NotADirectoryError("Le cartelle fixed/ e moving/ devono esistere")
    
    dataset = DentalDataset(fixed_dir, moving_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    model = RegistrationModel().to(device)
    criterion = HybridLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    total_losses = []
    ncc_losses = []
    mse_losses = []
    reg_losses = []
    model.train()
    for epoch in range(400):
        total_loss = 0
        ncc_loss = 0
        mse_loss = 0
        reg_loss = 0
        for fixed, moving in loader:
            fixed = fixed.to(device)
            moving = moving.to(device)
            registered, flow = model(fixed, moving)
            loss, ncc, mse, reg = criterion(fixed, registered, flow)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            ncc_loss += ncc.item()
            mse_loss += mse.item()
            reg_loss += reg.item()
            batch_count = len(loader)
        total_losses.append(total_loss/batch_count)
        ncc_losses.append(ncc_loss/batch_count)
        mse_losses.append(mse_loss/batch_count)
        reg_losses.append(reg_loss/batch_count)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}')
    plot_loss_components(total_losses, ncc_losses, mse_losses, reg_losses)

    # Visualizzazione risultati e metriche
    with torch.no_grad():
        fixed, moving = next(iter(loader))
        fixed = fixed.to(device)
        moving = moving.to(device)
        registered, _ = model(fixed, moving)
        
        # Converti tutto sulla CPU per le metriche
        fixed_cpu = fixed.cpu()
        moving_cpu = moving.cpu()
        registered_cpu = registered.cpu()

        metrics_before = calculate_overlap_metrics(fixed_cpu, moving_cpu)
        metrics_after = calculate_overlap_metrics(fixed_cpu, registered_cpu)
        
        print("\nMetriche di sovrapposizione:")
        print(f"SSIM: {metrics_before['SSIM']:.4f} → {metrics_after['SSIM']:.4f}")
        print(f"PSNR: {metrics_before['PSNR']:.2f} → {metrics_after['PSNR']:.2f}")
        print(f"Dice: {metrics_before['Dice']:.4f} → {metrics_after['Dice']:.4f}")
        print(f"NCC: {metrics_before['NCC']:.3f} → {metrics_after['NCC']:.3f}")
        
        # Visualizzazione
        fixed_np = tensor2im(fixed_cpu)
        moving_np = tensor2im(moving_cpu)
        registered_np = tensor2im(registered_cpu)
        visualize_results(fixed_np, moving_np, registered_np, metrics_before, metrics_after)
        
        # Salvataggio
        registered_to_save = (registered_cpu + 1) / 2
        save_image(registered_to_save, 'registered.png')

if __name__ == '__main__':
    main()
