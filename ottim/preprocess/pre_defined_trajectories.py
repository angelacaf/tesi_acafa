import numpy as np
from projection.trajectory import Trajectory, Point
from scipy.stats import beta


def get_semi_circle_trajectory(cbct_matrix, process_info=None, args=None):
    field_size = list(np.shape(cbct_matrix)[:2])
    return None


def estimate_trajectory_by_beta_function(cbct_matrix, process_info=None, args=None):
    # Step 1: get MIP image
    field_size = list(np.shape(cbct_matrix)[:2])
    
    # define beta trajectory
    margin = int(field_size[0]*0.02)
    keypoints = [Point(margin, 0), Point(field_size[0]/2, field_size[1]), Point(field_size[0]-margin, 0)]
    start_point, end_point = keypoints[0], keypoints[-1]
    alpha_beta = 3.6 if process_info is None else getattr(process_info, 'alpha_beta', 3.6)  # misura ampiezza
    scaling = 100 if process_info is None else getattr(process_info, 'scaling', 100)  # misura altezza
    bias = 25 if process_info is None else getattr(process_info, 'bias', 25)  # misura spostamento verticale
    def curve_funct(xs):
        ys = field_size[1] - beta.pdf(xs/field_size[0], alpha_beta, alpha_beta) * scaling - bias
        return ys
    debug_info = {}
    return Trajectory(start_point, end_point, curve_funct), debug_info