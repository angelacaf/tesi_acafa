# This file is used to estimate the dental arch same as Oral-3D
import numpy as np
from scipy.signal import convolve2d as conv2d
import math
from skimage.morphology import skeletonize
from projection.trajectory import Trajectory, Point
from core.visualize import *
import pandas as pd
from preprocess.utils import get_axial_slices, generate_MIP
from scipy.interpolate import CubicSpline
import os
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import label

def get_lower_boundary(binary_mask):
    points = np.argwhere(binary_mask)
    # Raggruppa per riga (y)
    boundary = {}
    for y, x in points:  # Formato (riga, colonna)
        if y not in boundary or x < boundary[y]:
            boundary[y] = x
    # Crea immagine vuota e disegna il bordo
    result = np.zeros_like(binary_mask)
    for y, x in boundary.items():
        result[y, x] = 1

    return result


def get_upper_boundary(binary_mask):
    points = np.argwhere(binary_mask)
    # Raggruppa per riga (y)
    boundary = {}
    for y, x in points:  # Formato (riga, colonna)
        if y not in boundary or x > boundary[y]:
            boundary[y] = x
    # Crea immagine vuota e disegna il bordo
    result = np.zeros_like(binary_mask)
    for y, x in boundary.items():
        result[y, x] = 1
    
    return result

import scipy.io

def estimate_trajectory_by_dental_arch(cbct_matrix, prior_shape):
    # x, y della curva aggiornata
    x = prior_shape[:, 0]
    y = prior_shape[:, 1]

    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]

    if len(x_unique) >= 11:
        y_smoothed = savgol_filter(y_unique, window_length=11, polyorder=3)
    else:
        y_smoothed = y_unique

    base_spline = CubicSpline(x_unique, y_smoothed, bc_type='natural')

    n_ctrl_pts = 20
    x_ctrl = np.linspace(x_sorted[0], x_sorted[-1], n_ctrl_pts)
    y_ctrl = base_spline(x_ctrl)
    ctrl_points = list(zip(x_ctrl, y_ctrl))

    final_spline = CubicSpline(x_ctrl, y_ctrl, bc_type='natural')
    start_point = Point(x_ctrl[0], final_spline(x_ctrl[0]))
    end_point = Point(x_ctrl[-1], final_spline(x_ctrl[-1]))
    trajectory = Trajectory(start_point, end_point, final_spline)

    debug_info = {
        'skeleton': prior_shape,
        'spline_ctrl_points': ctrl_points
    }

    return trajectory, debug_info



def get_largest_component(mask):
    """
    Estrae la componente connessa più grande da una maschera binaria 3D.
    
    Args:
        mask (numpy.ndarray): Maschera binaria 3D (0 = sfondo, 1 = oggetto di interesse).
    
    Returns:
        numpy.ndarray: Maschera con solo la componente più grande.
    """
    labeled_array, num_features = label(mask)
    
    if num_features == 0:
        return mask  # Nessuna componente trovata
    
    # Conta il numero di voxel per ogni etichetta
    component_sizes = np.bincount(labeled_array.ravel())
    
    # Ignora il primo elemento (sfondo)
    largest_component_label = np.argmax(component_sizes[1:]) + 1
    
    # Crea una nuova maschera con solo la componente più grande
    largest_component = (labeled_array == largest_component_label).astype(np.uint8)
    
    return largest_component

def get_arch_mask(axial_mip, threshold, filt_it=1):
    # 3200 is best for tooth
    # 2200 is best for bone
    arch_mask = axial_mip > threshold
    arch_mask = smooth_mask(arch_mask, filt_it)
        # Mantieni solo la componente più grande
    arch_mask = get_largest_component(arch_mask)
    return arch_mask


def get_filter_kernel(filter_type, filter_shape, param):
    weights = np.zeros(shape=filter_shape)
    if filter_type == 'gaussian':
        h, w = filter_shape
        center_x = (w - 1) / 2
        center_y = (h - 1) / 2
        if param is None:
            sigma = np.sqrt(center_x ** 2 + center_y ** 2) * 4
        else:
            sigma = param['sigma']
        denominator = math.sqrt(2 * math.pi) * sigma
        for id_x in range(w):
            for id_y in range(h):
                distance = (id_x - center_x) ** 2 + (id_y - center_y) ** 2
                weight = math.exp(-distance / (2 * sigma ** 2)) / (denominator ** 2)
                weights[id_y, id_x] = weight
        weights = weights / sum(weights)

    elif filter_type == 'average':
        weights = np.ones(shape=filter_shape)
        weights = weights / np.sum(weights)
    else:
        raise ValueError('Unsupported filter type')
    return weights


def filt_image(image, filter_type, filter_shape, iteration=1, threshold=0.3, param=None):
    weights = get_filter_kernel(filter_type, filter_shape, param)
    filtered_image = image.copy()
    for i in range(iteration):
        filtered_image = conv2d(filtered_image, weights, mode='same') > threshold
    return filtered_image


def smooth_mask(arch_mask, filt_it):
    # remove noise
    arch_mask = filt_image(arch_mask, 'average', [20, 20], threshold=0.5, iteration=5) # [20, 20],

    # make mask smooth
    for i in range(filt_it):
        arch_mask = filt_image(arch_mask, 'gaussian', [15, 15], threshold=0.9)
        arch_mask = filt_image(arch_mask, 'average', [20, 20], threshold=0.85)
    return arch_mask


def get_valid_mean(img_matrix):
    mask = img_matrix > np.min(img_matrix)
    mean = np.average(img_matrix[mask])
    return mean


# merge points that have same x
def merge_points(sorted_point_list):
    merged_point_list = []
    first_point = sorted_point_list[0]
    x_pointer = first_point.x
    y_stack = [first_point.y]
    for point in sorted_point_list[1:]:
        if point.x == x_pointer:
            y_stack.append(point.y)
        else:
            merged_point_list.append(Point(x_pointer, np.mean(y_stack)))
            x_pointer = point.x
            y_stack = [point.y]
    merged_point_list.append(Point(x_pointer, np.mean(y_stack)))
    return merged_point_list


# get the skeleton of the mask image
def get_skeleton(binary_image):
    skeleton_image = skeletonize(binary_image)
    return skeleton_image


def get_keypoints_from_skeleton_image(skeleton_image, point_num=25):
    points_array = np.argwhere(skeleton_image == 1)
    # sort point by x
    points_array = points_array[np.argsort(points_array[:, 0])]
    points_list = [Point(point_value[0], point_value[1]) for point_value in points_array]
    
    # merge points
    points_list = merge_points(points_list)

    # sample points
    point_num = len(points_list)
    interval = (point_num - 1) / (point_num - 1)
    keypoints = [points_list[int(i*interval)] for i in range(point_num)]
    
    return keypoints
