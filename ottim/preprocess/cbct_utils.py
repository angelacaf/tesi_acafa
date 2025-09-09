# check each case if it get the correct panoramic_image image
from scipy.ndimage import rotate
import numpy as np
import torch
import pandas as pd

MAX_VOXEL_VALUE=3999

# get crop_info from info
def crop_case(cbct_matrix, process_info):
    s_max, c_max, a_max = np.shape(cbct_matrix)

    a_start, a_end = int(process_info['axial_start']), int(process_info['axial_end'])
    s_start, s_end = int(process_info['sagittal_start']), int(process_info['sagittal_end'])
    c_start, c_end = int(process_info['coronal_start']), int(process_info['coronal_end'])

    if s_end > s_max:
        pad_matrix = np.zeros([s_end, c_max, a_max])
        pad_matrix[:s_max, :, :] = cbct_matrix
        cbct_matrix = pad_matrix
        s_max = s_end

    if c_end > c_max:
        pad_matrix = np.zeros([s_max, c_end, a_max])
        pad_matrix[:, :c_max, :] = cbct_matrix
        cbct_matrix = pad_matrix

    return cbct_matrix[s_start:s_end, c_start:c_end, a_start:a_end]


def rotate_case(cbct_matrix, process_info):
    # positive for clockwise
    a_rot = process_info['axial_rot']
    a_rot = 0 if pd.isna(a_rot) else int(a_rot)

    # positive for anti-clockwise
    s_rot = process_info['sagittal_rot']
    s_rot = 0 if pd.isna(s_rot) else int(s_rot)

    # positive for anti-clockwise
    c_rot = process_info['coronal_rot']
    c_rot = 0 if pd.isna(c_rot) else int(c_rot)

    rotated_matrix = cbct_matrix

    if a_rot != 0:
        rotated_matrix = rotate(rotated_matrix, a_rot, axes=(1, 0), reshape=False)
    if s_rot != 0:
        rotated_matrix = rotate(rotated_matrix, s_rot, axes=(2, 1), reshape=False)
    if c_rot != 0:
        rotated_matrix = rotate(rotated_matrix, c_rot, axes=(2, 0), reshape=False)

    return rotated_matrix

def resize_case(cbct_matrix, new_shape, interpolation='trilinear'):
    w, h, d = np.shape(cbct_matrix)
    img_tensor = torch.tensor(cbct_matrix, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    img_tensor = torch.nn.functional.interpolate(img_tensor, new_shape, mode=interpolation)
    resized_image = img_tensor.squeeze().data.numpy()
    return resized_image


def normalize_case(cbct_matrix, max_voxel_value=MAX_VOXEL_VALUE):
    cbct_matrix = cbct_matrix - np.min(cbct_matrix)
    cbct_matrix[cbct_matrix > max_voxel_value] = max_voxel_value
    return cbct_matrix


def clamp_case(cbct_matrix, upper_bound=4000, lower_bound=None):
    if upper_bound is not None:
        cbct_matrix[cbct_matrix > upper_bound] = upper_bound
    if lower_bound is not None:
        cbct_matrix[cbct_matrix < lower_bound] = lower_bound
    return cbct_matrix
