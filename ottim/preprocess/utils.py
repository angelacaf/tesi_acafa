import numpy as np
from scipy.stats import norm
from projection.basic import Boundary, Point
from projection.trajectory import Trajectory
from projection.projection_ray import ProjectionRay
from core.tools import save_png, norm_theta
import math

MU_WATER = 0.206
MU_AIR = 0.0004


def get_axial_mip(cbct_matrix):
    axial_slices = get_axial_slices(cbct_matrix)
    axial_mip = generate_MIP(axial_slices, direction='axial')
    return axial_mip


def generate_MIP(img_matrix, direction='axial'):
    if direction == 'axial':
        return np.max(img_matrix, axis=2)
    elif direction == 'coronal':
        return np.max(img_matrix, axis=1)
    elif direction == 'sagittal':
        return np.max(img_matrix, axis=0)
    else:
        raise ValueError('Undefined direction')

# select axial slices uased to estimate dental arch
def get_axial_slices(cbct_matrix):
    coronal_mip = generate_MIP(cbct_matrix, direction='coronal')
    # show_gray(np.rot90(coronal_mip))

    # determine the tooth density distribution
    mu, std = get_tooth_dist(coronal_mip[coronal_mip>0])
    threshold = mu + 1.5 * std

    mask = coronal_mip >= threshold
    # show_gray(np.rot90(mask))

    coronal_dist = np.sum(mask, axis=0)
    idx_mean, idx_std = fit_gaussian(coronal_dist)

    start_idx = int(idx_mean - 1.8*idx_std)
    end_idx = int(idx_mean + 1.8*idx_std)

    # Assicuriamoci che gli indici siano validi
    depth = cbct_matrix.shape[2]  # Profondita  della CBCT

    #print(f"start_idx: {start_idx}, end_idx: {end_idx}, ProfonditÃ  disponibile: {depth}")

    # Correzione degli indici fuori limite
    start_idx = max(0, start_idx)
    end_idx = min(depth, end_idx)

    if start_idx >= end_idx:
        raise ValueError(f"Indici non validi! start_idx={start_idx}, end_idx={end_idx}, depth={depth}")

    axial_slices = cbct_matrix[:, :, start_idx:end_idx]

    return axial_slices


def fit_gaussian(pdf):
    peak_idx = np.argmax(pdf)
    sum_pdf = np.sum(pdf)  # Calcola la somma dei valori di pdf

    if sum_pdf > 0:  
        pdf = pdf / sum_pdf  # Normalizza solo se la somma Ã¨ maggiore di zero
    else:
        pdf[:] = 0  # Se la somma Ã¨ zero, evita la divisione e assegna 0 all'array

    std = 0
    for idx, pdf_item in enumerate(pdf):
        std += pdf_item*(idx-peak_idx)**2
    return peak_idx, np.sqrt(std)


def get_tooth_dist(volumes):
    volumes = volumes.reshape(-1)
    lower_bound = 0
    higher_bound = 4000
    volumes = volumes[np.logical_and(volumes >= lower_bound, volumes <= higher_bound)]
    mean, std = norm.fit(volumes)
    return mean, std


def get_projection_info_on_axial_mip(trajectory: Trajectory, axial_mip, sample_n=100, projection_distance=60, down_sampling=1):
    w, h = np.shape(axial_mip)
    image_boundary = Boundary(x_right=w, y_right=h)
    projection_rays = trajectory.sample_projection_rays(sample_n, ray_length=projection_distance)
    outside_curve = {'xs':[], 'ys':[], 'type': 'outter_bound'}
    inside_curve = {'xs':[], 'ys':[], 'type': 'inner_bound'}
    center_curve = {'xs':[], 'ys':[], 'type': 'trajectory'}
    projection_points_list = []
    
    for line_id, projection_ray in enumerate(projection_rays):
        if line_id % down_sampling != 0:
            continue
        sample_points = projection_ray.get_sample_points_within_distance_and_boundary(distance=projection_distance,
                                                                                      boundary=image_boundary, 
                                                                                      resolution=1)
        sample_num = len(sample_points)
        projection_points = {}
        projection_points['xs'] = [point.x for point in sample_points][:sample_num // 2]
        projection_points['ys'] = [point.y for point in sample_points][:sample_num // 2]
        projection_points['type'] = 'projection_ray'
        projection_points_list.append(projection_points)

        outside_curve['xs'].append(sample_points[0].x)
        outside_curve['ys'].append(sample_points[0].y)

        inside_curve['xs'].append(sample_points[-1].x)
        inside_curve['ys'].append(sample_points[-1].y)

        center_curve['xs'].append(projection_ray.center.x)
        center_curve['ys'].append(projection_ray.center.y)
        # add moved projection lines
        for delta_theta in [-math.pi/4]:
        # for delta_theta in []:
            moved_projection_ray = ProjectionRay(center = projection_ray.center, 
                                                 theta = norm_theta(projection_ray.theta + delta_theta),
                                                 length = int(projection_ray.length / math.cos(delta_theta)),
                                                 )
            add_sample_points = moved_projection_ray.get_sample_points_within_distance_and_boundary(distance=None,
                                                                                                    boundary=image_boundary,
                                                                                                    resolution=1)
            add_sample_num = len(add_sample_points)
            add_projection_points = {}
            add_projection_points['xs'] = [point.x for point in add_sample_points][:add_sample_num // 2]
            add_projection_points['ys'] = [point.y for point in add_sample_points][:add_sample_num // 2]
            add_projection_points['type'] = 'projection_ray'
            projection_points_list.append(add_projection_points)
    
    curve_info_list = projection_points_list + [outside_curve, inside_curve, center_curve]
    return curve_info_list

def get_projection_image_from_projection_rays(projection_rays, cbct_image, resolution=1, S=1200, mode='old'):
    image = []
    cbct_mu = get_mu_from_hu(cbct_image)
    w, h, d = np.shape(cbct_image)
    image_boundary = Boundary(x_right=w-1, y_right=h-1)

    for projection_ray in projection_rays:
        if projection_ray.length == 0:
            sample_points = projection_ray.get_sample_points_within_boundary(image_boundary, resolution)
        else:
            distance = projection_ray.length
            sample_points = projection_ray.get_sample_points_within_distance_and_boundary(distance, image_boundary, resolution)
        projection_slice = np.zeros(d)
        if mode == 'soft':
            for sample_point in sample_points:
                new_slice = np.copy(cbct_image[int(sample_point.x), int(sample_point.y), :])
                projection_slice += np.exp(new_slice / S)
        elif mode == 'hard':
            new_slice = cbct_mu[int(sample_point.x), int(sample_point.y), :]
            projection_slice += new_slice
        else:
            raise ValueError('Undefined render method: %s.' % mode)
        image.append(S * np.log(projection_slice))
    return np.array(image)


#oral_3d
def get_MPR_images_from_projection_rays(projection_rays, cbct_image, projection_distance, mode='old'):

    MPR_image = []
    w, h, d = np.shape(cbct_image)
    image_boundary = Boundary(x_right=w-1, y_right=h-1)

    for projection_ray in projection_rays:
        sample_points = projection_ray.get_sample_points_within_distance_and_boundary(
            distance=projection_distance,
            boundary=image_boundary,
            point_num=80
        )
        
        projection_slice = []
        for sample_point in sample_points:
            if image_boundary.is_point_inside(sample_point):
                new_slice = np.copy(cbct_image[int(sample_point.x), int(sample_point.y), :])
            else:
                new_slice = np.zeros(d, dtype=np.float32)
            projection_slice.append(new_slice)

        # Forza ogni fetta a contenere esattamente 80 punti
        if len(projection_slice) < 80:
            padding = [np.zeros(d, dtype=np.float32)] * (80 - len(projection_slice))
            projection_slice.extend(padding)
        elif len(projection_slice) > 80:
            projection_slice = projection_slice[:80]

        MPR_image.append(np.array(projection_slice))

    # Converti in array e fai lo swap degli assi: [80, N_rays, D]
    MPR_image = np.swapaxes(np.array(MPR_image), 0, 1)

    #print(f"[INFO] Shape finale MPR: {MPR_image.shape}")  # Debug utile

    return MPR_image


def get_projection_image_from_MPR_images(MPR_images, S=5):  
    projection_image = np.zeros_like(MPR_images[0])
    for MPR_image in MPR_images:
        projection_image += np.exp(MPR_image/S)
    projection_image = S * np.log(projection_image)
    return projection_image

def save_rendered_panoramic_image(px_img, save_path):
    # clamp
    
    mean = np.mean(px_img)
    std = np.std(px_img)
    threshold = 4
    px_img = (px_img - mean)/std
    px_img[px_img > threshold] = threshold
    px_img[px_img < -threshold] = -threshold
    
    # normalization
    px_img = (px_img - np.min(px_img))/(np.max(px_img) - np.min(px_img)) * 255
    px_img = np.array(px_img, dtype=np.uint8).transpose()
    save_png(px_img, save_path)
    

def move_projection_rays(projection_rays, theta_delta):
    moved_projection_rays = [ProjectionRay(center=projection_ray.center, 
                                          theta=norm_theta(projection_ray.theta - theta_delta),
                                          length = int(projection_ray.length/math.cos(theta_delta)),
                                         ) for projection_ray in projection_rays 
                            ]
    return moved_projection_rays

def get_augment_projection_rays(projection_rays, theta_range=math.pi/2, sample_num=12):
    resolution = theta_range / sample_num
    add_projection_rays = []
    for sample_id in range(sample_num + 1):
        if sample_id == sample_num / 2:
            continue
        delta_theta = -theta_range / 2 + resolution * sample_id
        moved_projection_rays = move_projection_rays(projection_rays, delta_theta)
        add_projection_rays += moved_projection_rays
    return add_projection_rays

def get_mu_from_hu(data_hu):
    mu = MU_WATER + data_hu / 1000 * (MU_WATER - MU_AIR)
    return mu

def get_hu_from_mu(data_mu):
    hu = 1000 * (data_mu - MU_WATER) / (MU_WATER - MU_AIR)
    return hu

def get_projection_zone_from_trajectory_by_distance(trajectory, field_size, distance_limit=40):
    points = [Point(x, y) for y in range(field_size[1]) for x in range(field_size[0])]
    points_distance = trajectory.get_distance_to_points(points)
    selection_list = points_distance<=distance_limit
    return [point for point, selection in zip(points, selection_list) if selection]

def convert_projection_rays_into_array(projection_lines):
    return np.array([projection_line.to_np_array() for projection_line in projection_lines])