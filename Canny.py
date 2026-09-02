from Conv2d.conv2d_simple import Conv2d
from GaussianFilter.GaussFilter import GaussianFilter
import numpy as np
import cv2

def GetGradients(img: np.ndarray):
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
    theta = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            gradients[i][j] = np.sqrt((Gx_img[i][j] ** 2) + (Gy_img[i][j] ** 2))
            theta[i][j] = np.arctan2(Gy_img[i][j], Gx_img[i][j])

    return gradients, theta

def NMS(gradients, theta):
    h, w = gradients.shape
    nms_result = np.zeros_like(gradients, dtype=np.float64)
    
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180.0
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            grad_val = gradients[i][j]
            
            if grad_val == 0:
                continue
            
            if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] < 180):
                neighbor1 = gradients[i][j-1]   # 左
                neighbor2 = gradients[i][j+1]   # 右
            
            elif 22.5 <= angle[i][j] < 67.5:
                neighbor1 = gradients[i-1][j+1]  # 右上
                neighbor2 = gradients[i+1][j-1]  # 左下
            
            elif 67.5 <= angle[i][j] < 112.5:
                neighbor1 = gradients[i-1][j]    # 上
                neighbor2 = gradients[i+1][j]    # 下
            
            elif 112.5 <= angle[i][j] < 157.5:
                neighbor1 = gradients[i-1][j-1]  # 左上
                neighbor2 = gradients[i+1][j+1]  # 右下
            
            if grad_val >= neighbor1 and grad_val >= neighbor2:
                nms_result[i][j] = grad_val
            else:
                nms_result[i][j] = 0
    
    return nms_result

img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gauss = GaussianFilter(img, 5, 5)

grad, theta = GetGradients(gauss)

res_grad = NMS(grad, theta)

h, w = res_grad.shape
result = res_grad.copy()
visited = np.zeros_like(res_grad)

thre1 = 20
thre2 = 30

def dfs(i, j):
    if i >= h or i < 0 or j >= w or j < 0 or visited[i][j] == 1:
        return
    visited[i][j] = 1
    if result[i][j] > thre2:
        result[i][j] = 255
        dfs(i-1, j-1)
        dfs(i-1, j)
        dfs(i-1, j+1)
        dfs(i, j-1)
        dfs(i, j+1)
        dfs(i+1, j-1)
        dfs(i+1, j)
        dfs(i+1, j+1)
    else:
        pass

for i in range(h):
    for j in range(w):
        if visited[i][j] == 1:
            continue
        if result[i][j] >= thre2:
            dfs(i, j)
        elif result[i][j] <= thre1:
            result[i][j] = 0
            visited[i][j] = 1

for i in range(h):
    for j in range(w):
        if visited[i][j] == 0:
            result[i][j] = 0
        elif result[i][j] < thre2 and result[i][j] > thre1:
            result[i][j] = 255

cv2.imshow('', result)
cv2.waitKey(0)