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
from skimage.metrics import structural_similarity as ssim
import numpy as np
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
from functools import partial
CANNY_LOW = 100
CANNY_HIGH = 200

def obj_surface_func(SPL, RT):
    return 0.005 * (SPL - np.mean(SPL))**2 + 0.002 * (RT - np.mean(RT))**2 + 0.1

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



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
    print(f"\n[INFO] Top {top_n} curve più distanti dalla media:")
    for fname, err in worst_cases:
        print(f"  {fname:<40} RMSE = {err:.2f}")
    df = pd.DataFrame(avg_ctrl_points, columns=["x", "y"])
    df.to_csv("avg_ctrl_points.csv", index=False)
    print("\nPunti di controllo medi:")
    print(df)
    
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

def reduce_to_control_points(prior_shape, num_ctrl_pts=20):
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

from scipy.signal import correlate2d

def normalized_cross_correlation(img1, img2):
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)
    numerator = np.sum(img1 * img2)
    denominator = np.sqrt(np.sum(img1**2) * np.sum(img2**2)) + 1e-8
    return numerator / denominator

from sklearn.metrics import mutual_info_score

def mutual_information(img1, img2, bins=64):
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)  # marginale x
    py = np.sum(pxy, axis=0)  # marginale y
    px_py = np.outer(px, py)
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / (px_py[nzs] + 1e-8)))


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
from scipy.optimize import least_squares, dual_annealing

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

def get_dice(gt_img, pr_img):
    mask_1 = gt_img > 0
    mask_2 = pr_img > 0
    dice = 2 * np.sum(np.logical_and(mask_1, mask_2)) / (np.sum(mask_1) + np.sum(mask_2)) 
    return dice*100

def convert_to_oral_3d_px(px_img):
    mean = np.mean(px_img)
    std = np.std(px_img)
    threshold = 4
    px_img = (px_img - mean)/std
    px_img[px_img > threshold] = threshold
    px_img[px_img < -threshold] = -threshold
    
    px_img = (px_img - np.min(px_img))/(np.max(px_img) - np.min(px_img)) * 255
    return np.array(px_img, dtype=np.uint8)

def apply_raytracing_params(spline_curve, cbct_volume, tx=0.0, ty=0.0, angle_deg=0.0, proj_dist=80.0, scale=1.0):
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
    px_sim= normalize_image(px_sim)
    return px_sim.T  

def plot_loss_components(edge_loss, ncc_loss, total_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(edge_loss, label="Edge Loss (weighted)", color='orange')
    plt.plot(ncc_loss, label="NCC Loss (1 - NCC)", color='blue')
    plt.plot(total_loss, label="Total Combined Loss", color='red', linestyle='--')
    plt.xlabel("Step di ottimizzazione")
    plt.ylabel("Valore della Loss")
    plt.title("Contributi alla Loss durante l'Ottimizzazione")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
def interpolate_spline_from_control_points(control_points, num_points=576):
    x, y = control_points[:, 0], control_points[:, 1]
    tck, _ = splprep([x, y], s=0)
    u = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u, tck)
    return np.stack([x_new, y_new], axis=1)

def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def normalize_volume_minmax(volume):
    return (volume - volume.min()) / (volume.max() - volume.min())
def rescale_to_cbct_range(volume, target_min=0, target_max=3999):
    vol_norm = normalize_volume_minmax(volume)
    return vol_norm * (target_max - target_min) + target_min
def show_single_projection(spline_curve, cbct_volume, tx=0.0, ty=0.0, angle_deg=0.0,
                           scale=1.0, proj_dist=60.0, title=None):
    px_sim = apply_raytracing_params(spline_curve, cbct_volume, tx=tx, ty=ty,
        angle_deg=angle_deg, scale=scale, proj_dist=proj_dist)












# === Configurazioni ===
case_id = 1
device = 'cuda:0'
ckpt_path = './model_best.pth.tar'
base_path = f'./case_{case_id:03d}.mat'
avg_curve_mat = './avg_curve.mat'
mat_path = f'./output_GENERATED/case_{case_id:03d}_generated.mat'
output_dir = './output_GENERATED'

# === Caricamento dati ===
data = scipy.io.loadmat(base_path)
px_real = data['Ideal_PX'].astype(np.float32)
cbct_true = data['CBCT']

# === Controllo iniziale spline ===
avg_data = scipy.io.loadmat(avg_curve_mat)
control_points_init = avg_data['AvgSplineCtrlPoints']
fixed_spline = interpolate_spline_from_control_points(control_points_init, num_points=576)

# === Modello Oral3D ===
px_tensor = torch.tensor(px_real, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
model = Oral3D(device=device)
model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
model.to(device)
model.eval()

with torch.no_grad():
    mpr_volume = model.generate(px_tensor, VAL=False).squeeze().cpu().numpy()
    mpr_volume = np.transpose(mpr_volume, (2, 0, 1))


#  generare un volume CBCT iniziale prima dell'ottimizzazione, confrontare la curva media con la curva ottimizzata
cbct_volume = interpolate_cbct(mpr_volume, fixed_spline)
cbct_volume = rescale_to_cbct_range(cbct_volume)
cbct_true = (cbct_true - cbct_true.min()) / (cbct_true.max() - cbct_true.min())
cbct_volume = (cbct_volume - cbct_volume.min()) / (cbct_volume.max() - cbct_volume.min())
cbct_volume = normalize_case(cbct_volume)
cbct_true = normalize_case(cbct_true)

# ➜  Opzionale: perturbazione random sui punti di controllo iniziali
CP_PERTURB_SIGMA = 3.0   # px; impostare 0.0 per disattivare

pert = np.random.normal(scale=CP_PERTURB_SIGMA, size=control_points_init.shape)
ctrl_init_k = control_points_init + pert
params_spline_init = ctrl_init_k.flatten()
params_rt_init     = np.array([0.0,0.0,0.0,1.0,80.0])  # tx ty θ scale dist

# --- bounds spline ---------------------------------------------
x_min,x_max = 0,287; y_min,y_max = 0,255
xC = params_spline_init[::2]; yC = params_spline_init[1::2]
low_spl = np.empty_like(params_spline_init); up_spl = np.empty_like(params_spline_init)
low_spl[::2] = np.clip(xC-30,x_min,x_max); up_spl[::2] = np.clip(xC+30,x_min,x_max)
low_spl[1::2] = np.clip(yC-50,y_min,y_max); up_spl[1::2] = np.clip(yC+50,y_min,y_max)

# --- bounds RT --------------------------------------------------
low_rt = np.array([-10,-10,-10,0.9,40]); up_rt = np.array([10,10,10,1.1,110])

# === Parametri globali ===
params_init_full  = np.concatenate([params_spline_init, params_rt_init])
lower_bounds_full = np.concatenate([low_spl, low_rt])
upper_bounds_full = np.concatenate([up_spl, up_rt])

# ===============================================================
# 4) HYPER‑PARAMS OPTIMIZER --------------------------------------
# ===============================================================
OPTIMIZER_MODE   = "lsq"   # "lsq" | "sa"
LOSS_MODE        = "soft_l1"
F_SCALE          = 0.1
DIFF_STEP        = 1e-2
USE_ANALYTIC_JAC = False
SA_MAX_ITER      = 1200
SMOOTH_DSSIM_N   = 5

# ===============================================================
# 5) TRACKER GLOBALI
# ===============================================================
step_cnt       = 0
param_track    = []
gradient_track = []
dssim_history  = []

# ===============================================================
# 6) FUNZIONI COST & SUPPORT ------------------------------------
# ===============================================================

def compute_dssim_mean(px_sim, px_real):
    px_sim_n  = (px_sim - px_sim.min())/(px_sim.max()-px_sim.min()+1e-8)
    px_real_n = (px_real-px_real.min())/(px_real.max()-px_real.min()+1e-8)
    _, ssim_map = ssim(px_real_n, px_sim_n, data_range=1.0, full=True)
    dssim_map = (1-ssim_map)/2.0
    return dssim_map.mean(), dssim_map.flatten()


def residuals_full(params, cbct_vol, px_real, metric='DSSIM'):
    global step_cnt
    step_cnt += 1
    n_ctrl = control_points_init.shape[0]
    ctrl = params[:2*n_ctrl].reshape(-1, 2)
    tx, ty, θ, scale, dist = params[2*n_ctrl:]

    spline = interpolate_spline_from_control_points(ctrl)
    px_sim = apply_raytracing_params(spline, cbct_vol, tx=tx, ty=ty,
                                     angle_deg=θ, scale=scale, proj_dist=dist).astype(np.float32)
    px_sim = convert_to_oral_3d_px(px_sim)

    # Normalizza per NCC e MI
    px_sim_norm = normalize_image(px_sim)
    px_real_norm = normalize_image(px_real)

    # === Scelta della metrica ===
    if metric.upper() == 'DSSIM':
        dssim_mean, resid = compute_dssim_mean(px_sim_norm, px_real_norm)
        loss_value = dssim_mean

    elif metric.upper() == 'NCC':
        score = normalized_cross_correlation(px_sim_norm, px_real_norm)
        resid = 1-score  # Per minimizzare
        loss_value = resid

    elif metric.upper() == 'MI':
        score = mutual_information(px_sim_norm, px_real_norm)
        resid = 1-score  # Per minimizzare
        loss_value = resid

    else:
        raise ValueError(f"Metrica '{metric}' non supportata. Usa 'DSSIM', 'NCC' o 'MI'.")

    # Logging
    dssim_history.append(loss_value)
    param_track.append(params.copy())
    if len(param_track) >= 2:
        gradient_track.append(-(param_track[-1] - param_track[-2]))

    print(f"[iter {step_cnt:03d}] {metric} = {loss_value:.5f}")
    return resid


def objective_sa(params, cbct_vol, px_real):
    n_ctrl = control_points_init.shape[0]
    ctrl = params[:2*n_ctrl].reshape(-1,2)
    tx,ty,θ,scale,dist = params[2*n_ctrl:]
    spline = interpolate_spline_from_control_points(ctrl)
    px_sim = apply_raytracing_params(spline, cbct_vol, tx=tx, ty=ty,
                                     angle_deg=θ, scale=scale, proj_dist=dist).astype(np.float32)
    px_sim = convert_to_oral_3d_px(px_sim)
    dssim_mean,_ = compute_dssim_mean(px_sim, px_real)
    dssim_history.append(dssim_mean); param_track.append(params.copy())
    return dssim_mean


# ===============================================================
# 7) OPTIMIZATION DRIVER ----------------------------------------
# ===============================================================
if OPTIMIZER_MODE=="lsq":
    res = least_squares(residuals_full, params_init_full,
                        bounds=(lower_bounds_full, upper_bounds_full),
                        args=(cbct_volume, px_real), method="trf",
                        loss=LOSS_MODE, f_scale=F_SCALE, diff_step=DIFF_STEP,
                        jac="2-point" if not USE_ANALYTIC_JAC else jacobian_full,
                        max_nfev=800, verbose=2, gtol=1e-10, ftol=1e-10)
    best_params = res.x
else:
    best_params = dual_annealing(lambda p: objective_sa(p, cbct_volume, px_real),
                                 bounds=list(zip(lower_bounds_full, upper_bounds_full)),
                                 maxiter=SA_MAX_ITER, no_local_search=True).x
    
# --------------------------------------------------------
# 7.2) Stampa dei parametri di proiezione ottimizzati
# --------------------------------------------------------
n_ctrl = control_points_init.shape[0]
tx, ty, theta_deg, scale, proj_dist = best_params[2 * n_ctrl:]

print("\n===== Parametri di Proiezione Ottimizzati =====")
print(f"Traslazione X   : {tx:.3f} px")
print(f"Traslazione Y   : {ty:.3f} px")
print(f"Rotazione       : {theta_deg:.3f}°")
print(f"Scala           : {scale:.4f}")
print(f"Distanza Focale : {proj_dist:.2f} px")
print("================================================")

# ------------------------------------------------------------------
# 7.1)  SSIM PRE vs POST OTTIMIZZAZIONE ---------------------------
# ------------------------------------------------------------------

def project_px(params):
    """Ricalcola la proiezione PX dato un vettore-parametri completo."""
    n_ctrl = control_points_init.shape[0]
    ctrl   = params[:2*n_ctrl].reshape(-1,2)
    tx,ty,th,sc,dist = params[2*n_ctrl:]
    spline = interpolate_spline_from_control_points(ctrl)
    px_sim = apply_raytracing_params(spline, cbct_volume,
                                     tx=tx, ty=ty,
                                     angle_deg=th, scale=sc, proj_dist=dist)
    return convert_to_oral_3d_px(px_sim)

_norm = lambda x:(x-x.min())/(x.max()-x.min()+1e-8)
px_init  = project_px(params_init_full)
px_final = project_px(best_params)
ssim_init,_  = ssim(_norm(px_real), _norm(px_init),  data_range=1.0, full=True)
ssim_final,_ = ssim(_norm(px_real), _norm(px_final), data_range=1.0, full=True)
print("===============  SSIM SUMMARY  ===============")
print(f"SSIM iniziale : {ssim_init:.4f}")
print(f"SSIM finale   : {ssim_final:.4f}")
print("============================================")


# ------------------------------------------------------------------
# 7.2) SSIM VOLUMETRICA (CBCT vs CBCT ottimizzata)
# ------------------------------------------------------------------

from utils.ssim import SSIM  # se non l’hai già importato altrove

# 1. Normalizzazione volumi CBCT
cbct_true_norm = normalize_case(cbct_true)

# --- Volume interpolato iniziale (da spline perturbata)
ctrl_init = params_init_full[:2*control_points_init.shape[0]].reshape(-1,2)
spline_init = interpolate_spline_from_control_points(ctrl_init)
cbct_init = normalize_case(rescale_to_cbct_range(interpolate_cbct(mpr_volume, spline_init)))

# --- Volume interpolato ottimizzato (da best_params)
ctrl_final = best_params[:2*control_points_init.shape[0]].reshape(-1,2)
spline_final = interpolate_spline_from_control_points(ctrl_final)
cbct_final = normalize_case(rescale_to_cbct_range(interpolate_cbct(mpr_volume, spline_final)))

# 2. SSIM volumetrica
ssim_model = SSIM(device=device)
ssim_vol_init  = ssim_model.eval_ssim(cbct_true_norm, cbct_init)
ssim_vol_final = ssim_model.eval_ssim(cbct_true_norm, cbct_final)

print("\n===== SSIM VOLUMETRICA =====")
print(f"CBCT iniziale vs true : {ssim_vol_init:.4f}")
print(f"CBCT ottimizzata vs true : {ssim_vol_final:.4f}")
print("============================\n")


# ===============================================================
# 8) PLOTTING ----------------------------------------------------
# ===============================================================

def plot_dssim(hist, window=SMOOTH_DSSIM_N, save=False):
    it = np.arange(1, len(hist) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(it, hist, "o-", label="raw")
    if window > 1 and len(hist) >= window:
        smooth = np.convolve(hist, np.ones(window) / window, mode="valid")
        plt.plot(it[window - 1 :], smooth, label=f"{window}-pt avg")
    plt.xlabel("Iterazione"); plt.ylabel("DSSIM medio"); plt.grid(alpha=0.3)
    plt.title("DSSIM raw+smoothed"); plt.legend(); plt.tight_layout()
    if save:
        plt.savefig("./dssim_curve.png", dpi=300)
    plt.show()


def plot_grad_arrow_original(grads, save=False):
    if len(grads)<2: return
    desc = np.vstack(grads); norms=np.linalg.norm(desc,axis=1)
    i_max=np.argmax(norms); i_min=np.argmin(norms[i_max:])+i_max
    start,end=desc[i_max][:3], desc[i_min][:3]
    ax=plt.figure(figsize=(6,5)).add_subplot(111, projection="3d")
    ax.quiver(*start, *(end-start), arrow_length_ratio=0.1, linewidth=2)
    ax.set_xlabel("Param1"); ax.set_ylabel("Param2"); ax.set_zlabel("Param3")
    ax.set_title("Gradient max→min (prime 3 dim)")
    rng=np.abs([start,end]).max()*1.2; ax.set_xlim([-rng,rng]); ax.set_ylim([-rng,rng]); ax.set_zlim([-rng,rng])
    plt.tight_layout()
    if save:
        plt.savefig("./grad_arrow.png", dpi=300)
    plt.show()

def plot_conceptual_arrow(param_list, dssim_hist, use_deltas=True, save=False):
    if not param_list: return
    n_ctrl = control_points_init.shape[0]
    spl = np.array([np.linalg.norm(p[:2*n_ctrl]) for p in param_list])
    rt  = np.array([np.linalg.norm(p[2*n_ctrl:]) for p in param_list])
    if use_deltas:
        spl -= spl[0]; rt -= rt[0]
    i_hi,i_lo = np.argmax(dssim_hist), np.argmin(dssim_hist)
    start = np.array([spl[i_hi], rt[i_hi], dssim_hist[i_hi]])
    end   = np.array([spl[i_lo], rt[i_lo], dssim_hist[i_lo]])
    ax = plt.figure(figsize=(6,5)).add_subplot(111,projection="3d")
    ax.quiver(*start, *(end - start), arrow_length_ratio=0.1, linewidth=2, color="navy")
    ax.set_xlabel("Δ‖Spline‖" if use_deltas else "‖Spline‖")
    ax.set_ylabel("Δ‖RT‖"      if use_deltas else "‖RT‖")
    ax.set_zlabel("DSSIM"); ax.set_title("Bacino SPL–RT–Obj")
    pad = 0.1 * np.abs(np.r_[start, end]).max()
    ax.set_xlim([start[0]-pad, end[0]+pad])
    ax.set_ylim([start[1]-pad, end[1]+pad])
    ax.set_zlim([0, end[2]+pad])
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    if save:
        plt.savefig("./conceptual_arrow.png", dpi=300)
    plt.show()

def plot_curve_comparison(ctrl_avg, ctrl_perturbed, ctrl_optim, save=False):
    curve_avg      = interpolate_spline_from_control_points(ctrl_avg)
    curve_pert     = interpolate_spline_from_control_points(ctrl_perturbed)
    curve_optim    = interpolate_spline_from_control_points(ctrl_optim)

    plt.figure(figsize=(6, 5))
    plt.plot(curve_avg[:, 0], curve_avg[:, 1], label="Curva Media (Avg)", linestyle='--', color='black')
    plt.plot(curve_pert[:, 0], curve_pert[:, 1], label="Curva Perturbata", linestyle=':', color='orange')
    plt.plot(curve_optim[:, 0], curve_optim[:, 1], label="Curva Ottimizzata", linestyle='-', color='red')
    plt.scatter(ctrl_avg[:, 0], ctrl_avg[:, 1], s=10, color='black')
    plt.scatter(ctrl_perturbed[:, 0], ctrl_perturbed[:, 1], s=10, color='orange')
    plt.scatter(ctrl_optim[:, 0], ctrl_optim[:, 1], s=10, color='red')
    plt.gca().invert_yaxis()
    plt.title("Confronto Curve Spline")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()

    if save:
        plt.savefig("./curve_comparison.png", dpi=300)
    plt.show()


def plot_px_pairs(px_real, px_init, px_final, ssim_init, ssim_final, save=False):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for a in ax: a.axis("off")
    ax[0].imshow(px_real, cmap="gray",origin='lower'); ax[0].set_title("PX reale")
    ax[1].imshow(px_init, cmap="gray", origin='lower'); ax[1].set_title(f"Init SSIM={ssim_init:.3f}")
    ax[2].imshow(px_final, cmap="gray",origin='lower'); ax[2].set_title(f"Final SSIM={ssim_final:.3f}")
    plt.tight_layout()
    if save:
        plt.savefig("./px_comparison.png", dpi=300)
    plt.show()

# ------------------------------------------------------------------
# 9) ESECUZIONE E SALVATAGGIO PLOT
# ------------------------------------------------------------------
ctrl_avg       = control_points_init
ctrl_perturbed = params_spline_init.reshape(-1, 2)
ctrl_optim     = best_params[:2 * control_points_init.shape[0]].reshape(-1, 2)

plot_curve_comparison(ctrl_avg, ctrl_perturbed, ctrl_optim, save=True)
from vedo import Volume, show

print("[INFO] Visualizzazione volume iniziale")
show(Volume(cbct_init, spacing=(1, 1, 1)), axes=1, viewup='z', interactive=True)
plot_dssim(dssim_history, save=True)
plot_grad_arrow_original(gradient_track, save=True)
plot_conceptual_arrow(param_track, dssim_history, save=True)
plot_px_pairs(px_real, px_init, px_final, ssim_init, ssim_final, save=True)
# ------------------------------------------------------------------
# 9) VISUALIZZAZIONE VOLUMETRICA (opzionale)
# ------------------------------------------------------------------


print("[INFO] Visualizzazione volume ottimizzato")
show(Volume(cbct_final, spacing=(1, 1, 1)), axes=1, viewup='z', interactive=True)
