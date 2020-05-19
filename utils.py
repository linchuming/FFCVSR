import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2

def psnr(im1, im2):
    """ im1 and im2 value must be between 0 and 255"""
    im1 = np.float64(im1)
    im2 = np.float64(im2)
    rmse = np.sqrt(np.mean(np.square(im1[:] - im2[:])))
    psnr = 20 * np.log10(255 / rmse)
    return psnr, rmse


def img_to_uint8(img):
    img = np.clip(img, 0, 255)
    return np.round(img).astype(np.uint8)


rgb_to_ycbcr = np.array([[65.481, 128.553, 24.966],
                         [-37.797, -74.203, 112.0],
                         [112.0, -93.786, -18.214]])

ycbcr_to_rgb = np.linalg.inv(rgb_to_ycbcr)


# ycbcr_to_rgb = np.array([[1.164, 0, 1.596],
#                          [1.164, -0.813, -0.392],
#                          [1.164, 2.017, 0]])

def rgb2ycbcr(img):
    """ img value must be between 0 and 255"""
    img = np.float64(img)
    img = np.dot(img, rgb_to_ycbcr.T) / 255.0
    img = img + np.array([16, 128, 128])
    return img


def ycbcr2rgb(img):
    """ img value must be between 0 and 255"""
    img = np.float64(img)
    img = img - np.array([16, 128, 128])
    img = np.dot(img, ycbcr_to_rgb.T) * 255.0
    return img

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')