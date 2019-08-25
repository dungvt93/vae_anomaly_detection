#!/usr/bin/env python
import sys
from zipfile import ZipFile
from PIL import Image # $ pip install pillow
import numpy as np
import cv2
import os

dir = "../2019-08-19/"
num_data = 300
for folder in os.listdir(dir):
    with ZipFile(dir+folder) as archive:
        if len(sys.argv) > 1 and sys.argv[1] == "random":
            for i in range(num_data):
                j = np.random.randint(0,len(archive.infolist()))
                entry = archive.infolist()[j]
                with archive.open(entry) as file:
                    img = Image.open(file)
                    img = np.array(img)
                    img = img[:, :, ::-1].copy()
                    cv2.imwrite('../20190819_dataset/origin/'+ os.path.basename(file.name),img)
        else:
            for i in range(len(archive.infolist())):
                entry = archive.infolist()[i]
                with archive.open(entry) as file:
                    img = Image.open(file)
                    img = np.array(img)
                    img = img[:, :, ::-1].copy()
                    cv2.imwrite('../20190819_dataset/origin/'+ os.path.basename(file.name),img)
