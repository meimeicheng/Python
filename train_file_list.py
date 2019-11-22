# -*- coding: utf-8 -*-
"""
將要訓練的圖片路徑讀取成txt格式
"""

import os

path = "./dogs/"
file_ext = (".jpg", ".gif", ".png")

for filenames in os.walk(path):
    filenames = list(filenames)
    filepath = filenames[0]
    filenames = filenames[2]
    for filename in filenames:
        print(filename)
        if filename.lower().endswith(tuple(file_ext)):
            with open ("../YOLOv3/data/dataset/dog_train.txt", 'a') as f:
                f.write(filepath+'/'+filename+'\n')
