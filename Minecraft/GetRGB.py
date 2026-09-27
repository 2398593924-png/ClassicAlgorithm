import cv2
import os
import json
import numpy as np


path = "./Minecraft/color_block"
output_path = "./Minecraft/color_table.json"

data = {}

for f in os.listdir(path):
    img = cv2.imread(os.path.join(path, f))
    rgb_avg = []
    for i in range(3):
        rgb_avg.append(int(np.mean(img[:, :, i])))
    
    data[f[:-4]] = rgb_avg

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
