import os
import sys
import cv2
import numpy as np
import zipfile
from PIL import Image
import tnn
import vae_16x16_tf

def inference_zip_folder(zip_folder,des_folder="./result"):
    if not os.path.exists(des_folder):
        os.mkdir(des_folder)

    zip_folder_name = os.path.basename(os.path.splitext(zip_folder)[0])
    if not os.path.exists(des_folder + "/" + zip_folder_name):
        os.mkdir(des_folder + "/" + zip_folder_name)

    with zipfile.ZipFile(zip_folder) as archive:
        for entry in archive.infolist():
            with archive.open(entry) as file:
                img = Image.open(file)
                img = np.array(img)
                #convert RGB to BGR
                img = img[:,:,::-1]
                _,_,result = tnn_model.detect(img,detect_threshold)
                if len(result) != 0 and result[0] is not None:
                    result_img = run_judgement_model(result[0],judgement_threshold)
                    if result_img is not None:
                        print(des_folder+"/"+zip_folder_name+"/"+os.path.basename(file.name))
                        print(cv2.imwrite(des_folder+"/"+zip_folder_name+"/"+os.path.basename(file.name),result_img))
                else:
                    print("can't detect any pin")

def inference_folder(input_folder):
    for file_name in os.listdir(input_folder):
        img = cv2.imread(input_folder + "/" + file_name)
        _,_,result = tnn_model.detect(img,detect_threshold)
        if len(result) != 0:
            result_img = run_judgement_model(result[0],judgement_threshold)
            if result_img is not None:
                print(des_folder + file_name)
                print(cv2.imwrite(des_folder + file_name),img)
        else:
            print("can't detect any pin")

def run_judgement_model(test_img,threshold):
    result_img = None
    input_normal = cv2.imread(root_direct + 'compare.png')
    # list_input_anomaly = [cv2.imread(directory + file_name)]
    list_input_anomaly = [test_img]
    for result in  judgement_model.detect(input_normal,list_input_anomaly):
        img = cv2.resize(list_input_anomaly[0],(judgement_model.img_shape[1],judgement_model.img_shape[0]))
        for idx,score in enumerate(result):
            y_ano,x_ano = np.where(score > threshold)
            if len(x_ano) != 0:
                for x,y in zip(x_ano,y_ano):
                    result_img = cv2.rectangle(img,(x*8,idx * 64 + y*8),(x*8+8,idx * 64 + y*8+8),(0,255,0),1)

    return result_img

if __name__ == "__main__":
    root_direct =  "../20190819_dataset_pin_3/"

    detect_threshold = 0.8
    tnn_model = tnn.Model(root_direct)

    judgement_threshold = 9.5
    judgement_model = vae_16x16_tf.Model()
    judgement_model.load_model(root_direct + "model_vae_backup/model")

    input_folder = sys.argv[1]
    des_folder="./result/"

    if len(sys.argv) >= 3 and sys.argv[2] == "zip":
        if zipfile.is_zipfile(input_folder):
            inference_zip_folder(zip_folder=input_folder, des_folder=des_folder)
        elif os.path.isdir(input_folder):
            for zip_folder in os.listdir(input_folder):
                inference_zip_folder(zip_folder=input_folder+ "/" +zip_folder, des_folder=des_folder)
    else:
        inference_folder(input_folder)



