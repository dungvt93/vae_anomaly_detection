#!/usr/bin/env python
import sys
from zipfile import ZipFile
from PIL import Image # $ pip install pillow
import numpy as np
import cv2
import os

dir = "20190730/"
num_data = 5
for folder in os.listdir(dir):
    with ZipFile(dir+folder) as archive:
        for i in range(num_data):
            j = np.random.randint(0,len(archive.infolist()))
            entry = archive.infolist()[j]
            with archive.open(entry) as file:
                img = Image.open(file)
                img = np.array(img)
                img = img[:, :, ::-1].copy()
                cv2.imwrite('result/'+ os.path.basename(file.name),img)
