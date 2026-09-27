import cv2
import json
import numpy as np

table = None
with open("./Minecraft/color_table.json", "r", encoding="utf-8") as f:
    table = json.load(f)

def Get_Nearest(point_group, target):
    best_label = "black_concrete_powder"
    best_distance = np.linalg.norm(np.array(point_group[best_label]) - target)

    for label in point_group:
        cur_distance = np.linalg.norm(np.array(point_group[label]) - target)
        if cur_distance < best_distance:
            best_distance = cur_distance
            best_label = label
    
    return best_label

def Get_pixel_Map(origin_img, scale_ratio):
    h, w, _ = origin_img.shape
    n_h, n_w = h // scale_ratio, w // scale_ratio
    img = cv2.resize(origin_img, (n_h, n_w), interpolation=cv2.INTER_NEAREST)

    print(img.shape)

    result = [["" for _ in range(n_w)] for _ in range(n_h)]

    for r in range(n_h):
        for c in range(n_w):
            result[r][c] = Get_Nearest(table, np.array(img[c, r]))
    
    return np.array(result)

def Generate_Minecraft_Command(img, scale_ratio, top_left):
    pixel_map = Get_pixel_Map(img, scale_ratio)
    rows, cols = pixel_map.shape
    x0, y0, z0 = top_left

    with open("result.txt", "w", encoding="utf-8") as f:
        for r in range(rows):
            start = 0
            for c in range(1, cols + 1):
                if c == cols or pixel_map[r, c] != pixel_map[r, start]:
                    block = pixel_map[r, start]
                    if start == c - 1:
                        f.write(
                            f"/setblock {x0 + r} {y0} {z0 + start} {block}\n"
                        )
                    else:
                        f.write(
                            f"/fill {x0 + r} {y0} {z0 + start} "
                            f"{x0 + r} {y0} {z0 + c - 1} {block}\n"
                        )
                    start = c
    print("Done")

if __name__ == '__main__':
    img = cv2.imread('./test.png')
    Generate_Minecraft_Command(img, 3, (0, -60, 0))