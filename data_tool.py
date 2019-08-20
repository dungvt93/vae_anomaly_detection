import os
import cv2
import numpy as np
import sys
import shutil

if __name__ == "__main__":
    source_folder = "../20190808_dataset/temp/"
    des_folder = "../20190808_dataset/test/NG/"
    args = sys.argv
    list_sequences = []
    if len[args] < 2:
        print("please select delete or copy or move option at parameter")
    if args[1] == "delete":
        for idx,file_name in enumerate(sorted(os.listdir(source_folder))):
            if idx+1 in list_sequences:
                os.remove(source_folder+file_name)
                print(source_folder+file_name)
    elif args[1] == "copy":
        for idx,file_name in enumerate(sorted(os.listdir(source_folder))):
            if idx+1 in list_sequences:
                shutil.copyfile(source_folder+file_name,des_folder+file_name)
                print(source_folder+file_name)
    elif args[1] == "move":
        for idx,file_name in enumerate(sorted(os.listdir(source_folder))):
            if idx+1 in list_sequences:
                shutil.move(source_folder+file_name,des_folder+file_name)
                print(source_folder+file_name)