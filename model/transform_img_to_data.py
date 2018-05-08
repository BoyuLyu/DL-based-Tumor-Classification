import cv2
import os
import pandas as pd
from skimage import io
import numpy as np
from tqdm import tqdm

folder_path = '/Users/apple/Documents/Study/VT coursework/Computation for Life Sciences II/Project/Retrain_th1_th6/img_fold0/heatmap'
output_path = '/Users/apple/Documents/Study/VT coursework/Computation for Life Sciences II/Project/Retrain_th1_th6/img_fold0'
Condfidence = []

for i in tqdm(range(1, 34)):
    img_path = folder_path + '/' + str(i) + 'class_gray.png'
    im = io.imread(img_path)
    tmp = im - im.min()
    tmp_2 = tmp/tmp.max()
    tmp_3 = np.uint8(tmp_2 * 255.0)
    tmp_4 = tmp_3.flatten()
    Condfidence.append(tmp_4)

D = pd.DataFrame(Condfidence).transpose()
D.to_csv(os.path.join(folder_path, 'confidence.csv'), header=None, index=None)

