from Conv2d.conv2d_simple import Conv2d
from GaussianFilter.GaussFilter import GaussianFilter
from tqdm import tqdm
import numpy as np
import cv2

def GetEnergy(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    Gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    Gy = np.linalg.matrix_transpose(Gx)

    Gx_img = Conv2d(img, Gx)
    Gy_img = Conv2d(img, Gy)

    gradients = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            gradients[i][j] = np.abs(Gx_img[i][j]) + np.abs(Gy_img[i][j])

    return gradients

def find_min_seam(energy) -> list:
    h, w = energy.shape
    DP = np.zeros((h, w))
    DP[0] = energy[0]
    
    for i in range(1, h):
        for j in range(w):
            candidates = [DP[i-1, j]]
            if j > 0: candidates.append(DP[i-1, j-1])
            if j < w-1: candidates.append(DP[i-1, j+1])
            DP[i, j] = energy[i, j] + min(candidates)
    
    seam = np.zeros(h, dtype=int)
    seam[-1] = np.argmin(DP[-1])
    
    for i in range(h-2, -1, -1):
        prev = seam[i+1]
        cols = [prev]
        if prev > 0: cols.append(prev-1)
        if prev < w-1: cols.append(prev+1)
        seam[i] = min(cols, key=lambda c: DP[i, c])

    return seam

def SeamCarving(img: np.ndarray) -> np.ndarray:
    img_origin = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = GaussianFilter(img, 5, 1.6)
    h, w = img.shape

    energy = GetEnergy(img)
    seam = find_min_seam(energy)

    img_new = np.zeros((h, w - 1, 3), dtype=img_origin.dtype)
    offset_x = 0
    for i in range(len(seam)):
        for j in range(w - 1):
            if j == seam[i]: offset_x = 1
            img_new[i, j, :] = img_origin[i, j + offset_x, :]

    return img_new

if __name__ == '__main__':
    img = cv2.imread('branch.jpeg')
    for i in tqdm(range(100)):
        img = SeamCarving(img)
    
    cv2.imshow('', img)
    cv2.waitKey(0)