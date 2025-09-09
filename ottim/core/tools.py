import nibabel as nib
import os
import shutil
import numpy as np
import csv
import scipy.io as scio
from imageio import imread, imsave
import logging
from datetime import datetime
import math

def read_nii(nii_path):
    nii_file = nib.load(nii_path)
    return nii_file.get_fdata(), nii_file.affine

def save_nii(image_matrix, affine_matrix, file_path):
    nii_img = nib.Nifti1Image(image_matrix.astype(np.int16), affine_matrix)
    nib.save(nii_img, file_path)


def initialize_directory(dir_path, clear=True):
    """ The function to initialize an empty directory

    If it exists, remove all the files under it.
    """
    if os.path.exists(dir_path) and clear:
        clear_files_under_directory(dir_path)
    else:
        os.makedirs(dir_path, exist_ok=True)


def clear_files_under_directory(dir_path, prefix=None, postfix=None):
    """The function to clear files under the directory
    
    It could with specific prefix or postfix
    """
    files = list_directory(dir_path, prefix, postfix)
    for file in files:
        local_file = os.path.join(dir_path, file)
        if os.path.isdir(local_file):
            try:
                shutil.rmtree(local_file)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        else:
            os.remove(local_file)


def list_directory(path, prefix=None, postfix=None):
    """The funtion to list files under with postfix or prefix"""
    files = os.listdir(path)
    file_list_new = []
    for f in files:
        if postfix and not prefix:
            if f.endswith(postfix):
                file_list_new.append(f)
        elif prefix and not postfix:
            if f.startswith(prefix):
                file_list_new.append(f)
        elif prefix and postfix:
            if f.endswith(postfix) and f.startswith(prefix):
                file_list_new.append(f)
        else:
            if not f.startswith('.'):
                file_list_new.append(f)
    return sorted(file_list_new)


def load_csv(split_file):
    with open(split_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        data_list = {}
        for item in reader:
            data_list[item[0]] = item[1:]
    return data_list


# data_new dict into mat files for further use
def save_mat(mat_data, file_path):
    for key, value in mat_data.items():
        if value is None:
            raise ValueError('Find empty content in case_%s:%s' % (file_path, key))
    scio.savemat(file_path, mat_data, format='5', do_compression=True)

    
def read_mat(file_path):
    return scio.loadmat(file_path)


def save_png(image, save_path):
    # flip the image to make it look better
    # imsave(join_path(save_space, case_name), image)
    # pix_max = np.max(image)
    # pix_min = np.min(image)
    # if pix_max <= 10 or pix_min < 0 or pix_max > 256:
    #     image = (image - pix_min)/(pix_max - pix_min)
    #     image = image * 128
    #     image = image.astype(np.uint8)
    imsave(save_path, image)


def get_timestamp():
    date_time_obj = datetime.now()
    time_stamp = '%d-%02d-%02d_%02d-%02d-%02d' % (date_time_obj.year, date_time_obj.month, date_time_obj.day,
                                                  date_time_obj.hour, date_time_obj.minute, date_time_obj.second)
    return time_stamp

        
def initial_logger(log_dir, logger_file_name='default'):
    """Initial log with the standard template"""
    logger = logging.getLogger()
    os.makedirs(log_dir, exist_ok=True)
    # logger_format = logging.Formatter('[%(asctime)s]-[%(processName)s]-[%(threadName)s]-[%(levelname)s]: %(message)s')
    logger_format = logging.Formatter('[%(asctime)s]: %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(logger_format)
    # sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    if logger_file_name is None:
        pass
    else:
        logger_file_name = get_timestamp() + '.log' if logger_file_name == 'default' else logger_file_name
        fh = logging.FileHandler(os.path.join(log_dir, logger_file_name))
        fh.setFormatter(logger_format)
        logger.addHandler(fh)
        # print('Logging file at %s' % os.path.join(log_dir, logger_file_name))     
    
    logger.setLevel(logging.INFO)
    return logger

def log_args(logger, args):
    for arg, value in vars(args).items():
        logger.info("Argument %s: %r", arg, value)
        
def log_network_info(logger, network_name, network):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    logger.info("Number of parameters of %s: %i" % (network_name, num_params))

# normalize the theta into (-pi, pi]
def norm_theta(theta):
    n = (theta + math.pi) / (math.pi * 2)
    n = math.floor(n)
    theta = theta - math.pi * 2 * n
    if theta == -math.pi:
        theta = math.pi
    return theta