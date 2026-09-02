import numpy as np

from Conv2d.conv2d_simple import Conv2d

def GaussianFunction(sigma: float, x, y) -> np.float16:
    # Normalization Constant
    const = 1.0/(2 * np.pi * (sigma ** 2))
    # exp
    gauss_exp = np.exp(-((x ** 2) + (y ** 2))/(2 * (sigma ** 2)))
    
    return const * gauss_exp

def GetGaussianKernel(size: int, sigma: float) -> np.ndarray:
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i][j] = GaussianFunction(sigma, i - (size // 2), j - (size // 2))
    
    return matrix / np.sum(matrix)

def GaussianFilter(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    kernel = GetGaussianKernel(kernel_size, sigma)

    return Conv2d(img, kernel)