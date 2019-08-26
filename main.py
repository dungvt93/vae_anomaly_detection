import os
import sys
import cv2
import numpy as np
import zipfile
from PIL import Image
import tnn
import vae_16x16_tf
import time
import csv

def inference_zip_folder(zip_folder,des_folder="./result"):
    if not os.path.exists(des_folder):
        os.mkdir(des_folder)

    # zip_folder_name = os.path.basename(os.path.splitext(zip_folder)[0])
    # if not os.path.exists(des_folder + "/" + zip_folder_name):
    #     os.mkdir(des_folder + "/" + zip_folder_name)
    total_img = 0
    with zipfile.ZipFile(zip_folder) as archive:
        for entry in archive.infolist():
            total_img += 1
            with archive.open(entry) as file:
                start = time.time()
                img = Image.open(file)
                img = np.array(img)
                #convert RGB to BGR
                img = img[:,:,::-1]
                print("time of get from zip ",time.time() - start)
                _,_,result = tnn_model.detect(img,detect_threshold)
                if len(result) != 0 and result[0] is not None:
                    result_img, _ = run_judgement_model(result[0],judgement_threshold)
                    if result_img is not None:
                        print(des_folder+"/"+os.path.basename(file.name))
                        # print(des_folder+"/"+zip_folder_name+"/"+os.path.basename(file.name))
                        # print(cv2.imwrite(des_folder+"/"+zip_folder_name+"/"+os.path.basename(file.name),result_img))
                        print(cv2.imwrite(des_folder+"/"+os.path.basename(file.name),result_img))
                else:
                    print("can't detect any pin")
    return total_img

def inference_folder(input_folder):
    total_img = 0
    with open(des_folder + 'note.csv','w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        for root, dirs, files in os.walk(input_folder):
            for file_name in files:
                total_img += 1
                print(file_name)
                img = cv2.imread(input_folder + "/" + root + "/" + file_name)
                # img = cv2.imread(root_direct + "temp/20190819155943793991.png")
                _,_,result = tnn_model.detect(img,detect_threshold)
                if len(result) != 0 and result[0] is not None:
                    result_img, max_score = run_judgement_model(result[0],judgement_threshold)
                    if result_img is not None:
                        print(des_folder + file_name)
                        print(cv2.imwrite(des_folder + "/" + file_name,result_img))
                        print(cv2.imwrite(des_folder + "/origin/" + file_name,result[0]))
                        writer.writerow([des_folder + "/" + file_name, max_score])
                else:
                    print("can't detect any pin")
        return  total_img

def run_judgement_model(test_img,threshold):
    result_img = None
    max_score = 0
    # list_input_anomaly = [cv2.imread(directory + file_name)]
    list_input_anomaly = [test_img]
    for result in  judgement_model.detect(judgement_model.compare_img,list_input_anomaly):
        img = cv2.resize(list_input_anomaly[0],(judgement_model.img_shape[1],judgement_model.img_shape[0]))
        for idx,score in enumerate(result):
            y_ano,x_ano = np.where(score > threshold)
            if len(x_ano) != 0:
                max_score = np.max(score)
                for x,y in zip(x_ano,y_ano):
                    result_img = cv2.rectangle(img,(x*8,idx * 64 + y*8),(x*8+8,idx * 64 + y*8+8),(0,255,0),1)

    return result_img, max_score

if __name__ == "__main__":
    start = time.time()
    root_direct =  "../20190819_dataset_pin_3_left/"

    detect_threshold = 0.8
    tnn_model = tnn.Model(root_direct)

    judgement_threshold = 7.5
    judgement_model = vae_16x16_tf.Model(compare_img_path= root_direct + 'compare.png')
    judgement_model.load_model(root_direct + "model_vae/model")

    input_folder = sys.argv[1]
    des_folder=root_direct + "result/"
    if not os.path.exists(des_folder + "origin/"):
        os.mkdir(des_folder+"origin/")

    total = 0
    if len(sys.argv) >= 3 and sys.argv[2] == "zip":
        if zipfile.is_zipfile(input_folder):
            total += inference_zip_folder(zip_folder=input_folder, des_folder=des_folder)
        elif os.path.isdir(input_folder):
            for zip_folder in os.listdir(input_folder):
                total += inference_zip_folder(zip_folder=input_folder+ "/" +zip_folder, des_folder=des_folder)
    else:
        total += inference_folder(input_folder)

    print(total)
    print(time.time() - start)

