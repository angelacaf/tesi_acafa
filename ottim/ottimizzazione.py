import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from skimage.metrics import structural_similarity as ssim
from utils import *
import nibabel as nib
# === Logging losses ===
loss_history = []
edge_loss_history = []
ssim_loss_history = []
# === Funzione di costo combinata: Edge + SSIM ===
def residuals_function_edge_ssim(ctrl_pts_flat, px_real, generation_np, alpha=0.7):
    ctrl_pts = ctrl_pts_flat.reshape(-1, 2)
    spline_curve = interpolate_spline_from_control_points(ctrl_pts, num_points=576)
    cbct_volume = utilinterpolate_cbct(generation_np, spline_curve)

    # Ray tracing
    trajectory, _ = utils.estimate_trajectory_by_dental_arch(cbct_volume, spline_curve)
    rays = trajectory.sample_projection_rays(sample_n=576, ray_length=100)
    mpr_images = utils.get_MPR_images_from_projection_rays(rays, cbct_volume, projection_distance=60)
    px_sim = utils.get_projection_image_from_MPR_images(mpr_images).T

    # Edge Loss
    edge_sim = cv2.Canny(utils.to_uint8(px_sim), 50, 150).astype(np.float32)
    edge_real = cv2.Canny(utils.to_uint8(px_real), 50, 150).astype(np.float32)
    edge_residual = edge_sim - edge_real
    edge_loss = np.mean(np.abs(edge_residual))

    # SSIM Loss
    ssim_value = ssim(px_sim, px_real, data_range=px_real.max() - px_real.min())
    ssim_loss = 1 - ssim_value

    # Combinazione pesata
    total_loss = alpha * edge_loss + (1 - alpha) * ssim_loss

    # Logging
    edge_loss_history.append(edge_loss)
    ssim_loss_history.append(ssim_loss)
    loss_history.append(total_loss)

    print(f"[STEP] Edge = {edge_loss:.4f} | SSIM = {ssim_value:.4f} | Total = {total_loss:.4f}")

    return np.ones((10,)) * total_loss  # residui fittizi per least_squares

# === Plot Loss ===
def plot_losses():
    plt.figure(figsize=(10, 5))
    plt.plot(edge_loss_history, label='Edge Loss')
    plt.plot(ssim_loss_history, label='SSIM Loss')
    plt.plot(loss_history, label='Total Loss', linestyle='--')
    plt.xlabel('Iterazione')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Andamento della funzione di costo')
    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    # === Caricamento dati ===
    mat_path = './case_001.mat'  # Modifica se necessario
    mat_data = scipy.io.loadmat(mat_path)
    mpr_path = './output_GENERATED/MPR_case_001.nii.gz'
    mpr_nifti = nib.load(mpr_path)
    generation_np = mpr_nifti.get_fdata()
    px_real = mat_data['Ideal_PX'] if 'Ideal_PX' in mat_data else mat_data['PX_REAL']
    prior_shape = mat_data['AvgSplineCtrlPoints'] if 'AvgSplineCtrlPoints' in mat_data else mat_data['PriorShape']

    # === Inizializzazione controllo punti ===
    ctrl_pts_init = reduce_to_control_points(prior_shape, num_ctrl_pts=10)
    params_init = ctrl_pts_init.flatten()

    lower_bounds = params_init - 10
    upper_bounds = params_init + 10

    # === Ottimizzazione ===
    res = least_squares(
        residuals_function_edge_ssim,
        params_init,
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        max_nfev=200,
        xtol=1e-4,
        ftol=1e-4,
        gtol=1e-4,
        args=(px_real, generation_np, 0.7)
    )

    # === Plot finale ===
    plot_losses()
