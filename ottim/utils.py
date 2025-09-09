import argparse
import math
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import scipy.io
import imageio.v3 as iio
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from skimage.feature import canny
from tqdm import tqdm
from vedo import Line, Plotter, Volume, show
from scipy.spatial.distance import cdist
from skimage import exposure
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize

from skimage.metrics import structural_similarity as compare_ssim
import torch
import time
from datetime import timedelta

from core.tools import save_png
from core.visualize import *
from model.oral_3d import Oral3D
from preprocess.arch_estimate import estimate_trajectory_by_dental_arch
from preprocess.cbct_utils import *
from preprocess.pre_defined_trajectories import estimate_trajectory_by_beta_function
from preprocess.utils import get_MPR_images_from_projection_rays, get_projection_image_from_MPR_images
from preprocess.utils import *
from projection.basic import Point
from projection.projection_ray import ProjectionRay
from projection.trajectory import Trajectory
from utils.eval import get_psnr, get_dice
from utils.ssim import SSIM

CANNY_LOW = 100
CANNY_HIGH = 200

def load_all_control_points(mat_dir, ctrl_key='SplineCtrlPoints', n_ctrl_pts_expected=20):
    """
    Carica tutti i punti di controllo validi dai file .mat in una directory.

    Returns:
        all_ctrl_points (list of np.ndarray): Lista di array (20, 2)
        case_names (list of str): Nomi dei file .mat caricati correttamente
    """
    all_ctrl_points = []
    case_names = []

    for fname in os.listdir(mat_dir):
        if not fname.endswith('.mat'):
            continue
        data = scipy.io.loadmat(os.path.join(mat_dir, fname))
        if ctrl_key not in data:
            continue

        ctrl_pts = np.array(data[ctrl_key])
        if ctrl_pts.shape[1] != 2:
            ctrl_pts = ctrl_pts.T

        if ctrl_pts.shape[0] != n_ctrl_pts_expected:
            continue

        all_ctrl_points.append(ctrl_pts)
        case_names.append(fname)

    return all_ctrl_points, case_names

def compute_average_curve(all_ctrl_points, n_prior_pts=576):
    """
    Calcola la curva spline media a partire da un insieme di punti di controllo.

    Parametri:
    - all_ctrl_points: array di forma (N, M, 2), dove N = numero di casi, M = numero di punti di controllo.
    - n_prior_pts: numero di punti della curva interpolata finale (default: 576).

    Ritorna:
    - avg_ctrl_points: array di forma (M, 2), punti di controllo medi.
    - prior_shape: array di forma (n_prior_pts, 2), curva spline interpolata.
    """

    # === Calcola media su X e Y ===
    all_ctrl_points = np.array(all_ctrl_points)  # (N, 20, 2)
    avg_ctrl_points = np.mean(all_ctrl_points, axis=0)  # (20, 2) 
    x_avg, y_avg = avg_ctrl_points[:, 0], avg_ctrl_points[:, 1]

    # Costruisci la spline e genera la curva
    spline = CubicSpline(x_avg, y_avg, bc_type='natural')
    x_prior = np.linspace(x_avg.min(), x_avg.max(), n_prior_pts)
    y_prior = spline(x_prior)
    prior_shape = np.stack((x_prior, y_prior), axis=1)

    return avg_ctrl_points, prior_shape

def plot_and_save_average_with_outliers(avg_ctrl_points, prior_shape, all_ctrl_points, case_names, output_path='avg_curve.mat', top_n=5):
    """
    Salva i risultati medi in .mat, visualizza graficamente tutte le curve e stampa i casi più distanti dalla media.

    Args:
        avg_ctrl_points: array (20, 2)
        prior_shape: array (576, 2)
        all_ctrl_points: lista di curve originali (N, 20, 2)
        case_names: lista di nomi file (N)
        output_path: nome file per salvare .mat
        top_n: numero di outlier da mostrare
    """

    # === Salva curva media e spline in .mat ===
    scipy.io.savemat(output_path, {
        'AvgSplineCtrlPoints': avg_ctrl_points,
        'AvgPriorShape': prior_shape
    })

    # === Calcola distanza RMSE tra ogni curva e la media ===
    all_ctrl_points = np.array(all_ctrl_points)
    errors = np.sqrt(np.mean((all_ctrl_points - avg_ctrl_points[None, :, :])**2, axis=(1, 2)))

    # === Ordina per distanza decrescente ===
    sorted_idx = np.argsort(errors)[::-1]
    worst_cases = [(case_names[i], errors[i]) for i in sorted_idx[:top_n]]

    # === Plot ===
    plt.figure(figsize=(6, 5))
    for i, curve in enumerate(all_ctrl_points):
        color = 'red' if i in sorted_idx[:top_n] else 'gray'
        alpha = 0.8 if i in sorted_idx[:top_n] else 0.3
        plt.plot(curve[:, 0], curve[:, 1], color=color, alpha=alpha)

    plt.plot(avg_ctrl_points[:, 0], avg_ctrl_points[:, 1], 'bo-', label='Media X+Y')
    plt.plot(prior_shape[:, 0], prior_shape[:, 1], 'r--', label='Spline su 576 punti')
    plt.title("Spline media e curve originali")
    plt.xlabel("X (voxel)")
    plt.ylabel("Y (voxel)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Stampa outlier ===
    #print(f"\n[INFO] Top {top_n} curve più distanti dalla media:")
    #for fname, err in worst_cases:
    #    print(f"  {fname:<40} RMSE = {err:.2f}")


    #df = pd.DataFrame(avg_ctrl_points, columns=["x", "y"])
    #df.to_csv("avg_ctrl_points.csv", index=False)
    #print("\nPunti di controllo medi:")
    #print(df)
    
def save_nii(volume, affine_matrix, file_name, save_dir):
    nii_img = nib.Nifti1Image(volume, affine_matrix)
    os.makedirs(save_dir, exist_ok=True)
    nib.save(nii_img, os.path.join(save_dir, file_name))

def get_distance_to_curve(point, curve_pts):
    dists = np.linalg.norm(curve_pts - point, axis=1)
    min_idx = np.argmin(dists)
    sign = 1 if point[1] > curve_pts[min_idx, 1] else -1
    return min_idx, int(sign * dists[min_idx])

def interpolate_cbct(mpr, sample_pts):
    cbct = np.ones((288, 256, 160), dtype=np.float64) * -1
    for x in range(288):      
        for y in range(256):   
            min_id, dist = get_distance_to_curve(np.array([x, y]), sample_pts)
            slice_idy = dist + 40  # offset per centrare l'immagine
        # il volume viene riempito colonna per colonna, 
        # andando a recuperare valori lungo Z da MPR
            if 0 <= slice_idy < mpr.shape[1] and 0 <= min_id < mpr.shape[0]:
                #cbct[x, y, :] = mpr[min_id, slice_idy, :]  
                cbct[x, y, :] = mpr[min_id, mpr.shape[1] - 1 - slice_idy, :]

    return cbct

def interpolate_spline_from_control_points(control_points, num_points=576):
    x_ctrl, y_ctrl = control_points[:, 0], control_points[:, 1]
    tck, _ = splprep([x_ctrl, y_ctrl], s=0, k=min(3, len(control_points) - 1)) 
    u_interp = np.linspace(0, 1, num_points) 
    x_interp, y_interp = splev(u_interp, tck)
    return np.stack([x_interp, y_interp], axis=1)

def reduce_to_control_points(prior_shape, num_ctrl_pts=10):
    x, y = prior_shape[:, 0], prior_shape[:, 1] 
    tck, _ = splprep([x, y], s=0)   
    u_fine = np.linspace(0, 1, num_ctrl_pts) 
    x_new, y_new = splev(u_fine, tck) 
    return np.stack([x_new, y_new], axis=1) 

def auto_canny(image, sigma=0.33):
    """
    Applica Canny edge detection con soglie automatiche basate sul contrasto locale.
    - image: immagine in scala di grigi (può essere float o uint8)
    - sigma: parametro per controllare la sensibilità (default 0.33)
    """
    if image.dtype != np.uint8:
        image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
    
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def normalized_cross_correlation(im1, im2):
    im1 = (im1 - np.mean(im1)) / (np.std(im1) + 1e-8)   
    im2 = (im2 - np.mean(im2)) / (np.std(im2) + 1e-8)
    return np.mean(im1 * im2)

def to_uint8(image):
    image = np.nan_to_num(image)
    norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    return (norm * 255).astype(np.uint8)
def edge_dice_loss(px_sim, px_real):
    edge_sim = cv2.Canny(to_uint8(px_sim), CANNY_LOW, CANNY_HIGH)
    edge_real = cv2.Canny(to_uint8(px_real), CANNY_LOW, CANNY_HIGH)
    bin_sim = (edge_sim > 0).astype(np.uint8)
    bin_real = (edge_real > 0).astype(np.uint8)
    dice = get_dice(bin_sim, bin_real)
    return 1 - dice  # Dice loss

def show_edge_comparison(px_sim, px_real, title='Edge comparison'):
    px_sim_uint8 = to_uint8(px_sim)
    px_real_uint8 = to_uint8(px_real)
    edge_sim = cv2.Canny(px_sim_uint8, CANNY_LOW, CANNY_HIGH).astype(np.float32)
    edge_real = cv2.Canny(px_real_uint8, CANNY_LOW, CANNY_HIGH).astype(np.float32)

    # Binarizza le edge maps
    edge_sim_bin = (edge_sim > 0).astype(np.uint8)
    edge_real_bin = (edge_real > 0).astype(np.uint8)

    # Calcola il Dice coefficient
    dice = edge_dice_loss(px_sim_uint8, px_real_uint8)

    # Visualizza
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))
    axs[0].imshow(edge_real, cmap='gray', origin='lower')
    axs[0].set_title("Edge PX Reale")
    axs[1].imshow(edge_sim, cmap='gray', origin='lower')
    axs[1].set_title("Edge PX Sintetica")
    plt.suptitle(f"{title}\nDice = {dice:.4f}")
    plt.show()

    edge_residual = edge_sim - edge_real 
    loss_edge = np.mean(np.abs(edge_residual))


    print(f"[INFO] Dice = {dice:.4f}")
    return loss_edge

def show_weighted_area_on_px(px_image, title="Zona pesata per ottimizzazione"):
    plt.figure(figsize=(8, 4))
    plt.imshow(px_image, cmap='gray', origin='lower')
    plt.title(title)
    plt.axis("off")

    # Aggiungi rettangolo: (x, y, larghezza, altezza)
    rect = patches.Rectangle(
        (80, 30),         # (x=start_col, y=start_row)
        400,               # larghezza = 350 - 200
        90,                # altezza = 120 - 30
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    plt.gca().add_patch(rect)
    plt.show()

def get_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = np.max(img2)  # oppure 1.0 se già normalizzate
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def get_ssim(img1, img2):
    # img1 e img2 devono essere [H, W, D] oppure 2D slice-by-slice
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    ssim_total = 0
    count = 0
    for i in range(img1.shape[0]):
        ssim_slice = compare_ssim(img1[i], img2[i], data_range=img2[i].max() - img2[i].min())
        ssim_total += ssim_slice
        count += 1
    return ssim_total / count

def get_dice(pred, target):
    intersection = np.sum(pred * target)
    return 2.0 * intersection / (np.sum(pred) + np.sum(target) + 1e-8)

def convert_to_oral_3d_px(px_img):
    mean = np.mean(px_img)
    std = np.std(px_img)
    threshold = 4
    px_img = (px_img - mean)/std
    px_img[px_img > threshold] = threshold
    px_img[px_img < -threshold] = -threshold
    
    px_img = (px_img - np.min(px_img))/(np.max(px_img) - np.min(px_img)) * 255
    return np.array(px_img, dtype=np.uint8)

def apply_raytracing_params(spline_curve, cbct_volume, tx=0.0, ty=0.0, angle_deg=0.0, proj_dist=60.0, scale=1.0):
    trajectory, _ = estimate_trajectory_by_dental_arch(cbct_volume, spline_curve)

    rays = []
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    for ray in trajectory.sample_projection_rays(sample_n=576, ray_length=100):
        # Trasforma il centro del raggio
        pt = np.array([ray.center.x, ray.center.y])
        scaled = pt * scale
        rotated = R @ scaled                 # ruota il punto di partenza di un raggio
        transformed = rotated + np.array([tx, ty])
        ray.center.x = transformed[0]
        ray.center.y = transformed[1]

        ray.theta += angle_rad    # ruota il vettore direzionale del raggio

        rays.append(ray)


    mpr = get_MPR_images_from_projection_rays(rays, cbct_volume, projection_distance=proj_dist)
    px_sim = get_projection_image_from_MPR_images(mpr)

    return px_sim.T  

def show_single_projection(spline_curve, cbct_volume, tx=0.0, ty=0.0, angle_deg=0.0,
                           scale=1.0, proj_dist=60.0, title=None):
    px_sim = apply_raytracing_params(spline_curve, cbct_volume, tx=tx, ty=ty,
        angle_deg=angle_deg, scale=scale, proj_dist=proj_dist)

    plt.figure(figsize=(8, 6))
    plt.imshow(px_sim, cmap='gray')
    plt.title(title or f"TX={tx}, TY={ty}, Angle={angle_deg}°, Scale={scale}, Dist={proj_dist}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()