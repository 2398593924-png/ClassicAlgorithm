import numpy as np

def Padding(img: np.ndarray, padding_size: int) -> np.ndarray:
    if len(img.shape) == 3:
        h, w, c = img.shape
        new_img = np.zeros((h + 2 * padding_size, w + 2 * padding_size, c), dtype=np.float64)

        new_img[padding_size: padding_size + h, padding_size: padding_size + w, :] = img
    else:
        h, w = img.shape
        new_img = np.zeros((h + 2 * padding_size, w + 2 * padding_size), dtype=np.float64)

        new_img[padding_size: padding_size + h, padding_size: padding_size + w] = img

    return new_img

def Conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        h, w, c = img.shape
        new_img = np.zeros((h, w, c), dtype=np.float64)
    else:
        h, w = img.shape
        new_img = np.zeros((h, w), dtype=np.float64)

    offset = kernel.shape[0] // 2

    img_padding = Padding(img, offset)

    for i in range(h):
        for j in range(w):
            sub_img = img_padding[i:i + kernel.shape[0], j:j + kernel.shape[0]]
            if len(img.shape) == 3: 
                res = sub_img * kernel[:, :, None]
            else:
                res = sub_img * kernel


            new_img[i][j] = res.sum(axis=(0, 1))

    return new_img