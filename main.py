import threading
import time
import variational_autoencoder_16x16_rgb as vae_16
import numpy as np
import tensorflow as tf
from keras import backend as K


result = {}
result["position"] = 0
result["anomaly"] = False
class myThread (threading.Thread):
    def __init__(self, thread_id, model, x_normal, x_test):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.model = model
        self.x_normal = x_normal
        self.x_test = x_test

    def run(self):
        print("Bat dau " + self.name)
        with tf.Graph().as_default():
            self.model = vae_16.load_weights(self.model)
            anomaly_img,normal_img = vae_16.get_result_evaluate(self.model,self.x_normal,self.x_test,vae_16.input_shape[0], vae_16.input_shape[1], vae_16.move)
            sub_img = anomaly_img - normal_img
            if np.amax(sub_img) > 6:
                threadLock.acquire()
                result["position"] = self.thread_id
                result["anomaly"] = True
                threadLock.release()
                print(self.thread_id," running ")

base_img = '../lego/result/OK/000013.jpg.jpg'
test_img = '../lego/result/test/NG/011940.jpg.jpg'
# test_img = '../lego/result/OK/000013.jpg.jpg'
threadLock = threading.Lock()
threads = []
models = []
#load_model
for i in range(10):
    # sub_model = vae_16.load_weights("best")
    # models.append(sub_model)
    thread = myThread(i, "best",base_img,test_img)
    threads.append(thread)
# exit()
# for sub_model in models:
#     thread = myThread(i, sub_model,base_img,test_img)
#     threads.append(thread)

for t in threads:
    t.start()

for t in threads:
    t.join()
print ("Ket thuc Main Thread",result)