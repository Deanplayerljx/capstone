import numpy as np
import cv2
from tqdm import tqdm
import os
 
img_array = []
img_folder = 'first_30_origin'
for filename in tqdm(sorted(os.listdir(img_folder))):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(img_folder, filename))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

 
out_name = 'first_30_origin/result.avi'
out = cv2.VideoWriter(out_name,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
cv2.VideoWriter()
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
