import os
import cv2
import numpy as np
import sys
import shutil

if __name__ == "__main__":
    source_folder = "../20190830_dataset/20190906_result/origin/"
    compare_folder="../20190830_dataset/result/"
    des_folder = "../20190830_dataset/train_bonus/"
    args = sys.argv
    list_sequences = []
    if len(args) < 2:
        print("please select delete or copy or move option at parameter")
        exit()
    if args[1] == "delete":
        if len(list_sequences) != 0 :
            for idx,file_name in enumerate(sorted(os.listdir(source_folder))):
                if idx+1 in list_sequences:
                    os.remove(source_folder+file_name)
                    print(source_folder+file_name)
        else:
            des_folder_list = os.listdir(des_folder)
            for file_name in os.listdir(source_folder):
                if file_name in des_folder_list:
                    os.remove(source_folder+file_name)
                    print(source_folder+file_name)

    elif args[1] == "copy":
        if len(list_sequences) != 0 :
            for idx,file_name in enumerate(sorted(os.listdir(source_folder))):
                if idx+1 in list_sequences:
                    shutil.copyfile(source_folder+file_name,des_folder+file_name)
                    print(source_folder+file_name)
        else:
            compare_folder_list = os.listdir(compare_folder)
            for file_name in os.listdir(source_folder):
                if file_name in compare_folder_list:
                    shutil.copyfile(source_folder+file_name,des_folder+file_name)
                    print(source_folder+file_name)
    elif args[1] == "move":
        if len(list_sequences) != 0 :
            for idx,file_name in enumerate(sorted(os.listdir(source_folder))):
                if idx+1 in list_sequences:
                    shutil.move(source_folder+file_name,des_folder+file_name)
                    print(source_folder+file_name)
