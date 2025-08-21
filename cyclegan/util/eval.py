import numpy as np
import math


def get_psnr(img_1, img_2, SCALE=255.0):
    mse = np.mean((img_1 - img_2)**2)
    return 10 * np.log10((SCALE**2) / mse)

def get_iou(img_1, img_2):
    mask_1 = img_1 > 0
    mask_2 = img_2 > 0
    IOU = np.sum(np.logical_and(mask_1, mask_2)) / np.sum(np.logical_or(mask_1, mask_2))
    return IOU


def get_dice(gt_img, pr_img, threshold=127):
    mask_1 = gt_img > threshold
    mask_2 = pr_img > threshold
    dice = 2 * np.sum(np.logical_and(mask_1, mask_2)) / (np.sum(mask_1) + np.sum(mask_2))
    return dice * 100

def mse2distance(mse):
    return math.sqrt(mse)*math.sqrt(288**2+256**2)