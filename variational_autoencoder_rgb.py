from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape
from keras import optimizers
from keras.models import Model,model_from_json
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K
from keras.layers import BatchNormalization, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import load_model
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os,random
import cv2
import h5py

f = h5py.File('./file.h5', 'w')
# network parameters
input_shape=(8, 8, 3)
move = 4
batch_size = 128
latent_dim = 2
epochs = 10
Nc = 16
# dataset
data_shape = [86,86,3]

#ヒートマップの描画
def save_img(x_normal, x_anomaly, img_normal, img_anomaly, name):
    path = 'images/'
    if not os.path.exists(path):
        os.mkdir(path)

    #　※注意　評価したヒートマップを1～10に正規化
    img_max = np.max([img_normal, img_anomaly])
    img_min = np.min([img_normal, img_anomaly])
    img_normal = (img_normal-img_min)/(img_max-img_min) * 9 + 1
    img_anomaly = (img_anomaly-img_min)/(img_max-img_min) * 9 + 1
    #cv2.imwrite(path+'dkm.png',img_normal[0,:,:,0])
    f.create_dataset('a', data=img_anomaly[0,:,:,0]-img_normal[0,:,:,0])
    f.close()
    #exit()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(x_normal[0,:,:,0], cmap='gray')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(img_normal[0,:,:,0], cmap='Blues',norm=colors.LogNorm())
    plt.axis('off')
    plt.colorbar()
    plt.clim(1, 10)

    plt.title(name + "normal")

    plt.subplot(2, 2, 3)
    plt.imshow(x_anomaly[0,:,:,0], cmap='gray')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(img_anomaly[0,:,:,0]-img_normal[0,:,:,0], cmap='Blues',norm=colors.LogNorm())
    plt.axis('off')
    plt.colorbar()
    plt.clim(1, 10)

    plt.title(name + "anomaly")

    plt.savefig(path + name +".png")
    plt.show()
    plt.close()

#ヒートマップの計算
def evaluate_img(model, x_normal_path, x_anomaly_path, name, height=16, width=16, move=2):
    x_normal = cv2.imread(x_normal_path)
    x_normal = x_normal.reshape(1,data_shape[0],data_shape[1],data_shape[2])
    x_normal = x_normal / 255

    x_anomaly = cv2.imread(x_anomaly_path)
    x_anomaly = x_anomaly.reshape(1,data_shape[0],data_shape[1],data_shape[2])
    x_anomaly = x_anomaly / 255

    img_normal = np.zeros((x_normal.shape))
    img_anomaly = np.zeros((x_anomaly.shape))

    for i in range(int((x_normal.shape[1]-height)/move)):
        for j in range(int((x_normal.shape[2]-width)/move)):
            x_sub_normal = x_normal[0, i*move:i*move+height, j*move:j*move+width, :]
            x_sub_anomaly = x_anomaly[0, i*move:i*move+height, j*move:j*move+width, :]
            x_sub_normal = x_sub_normal.reshape(1, height, width, 3)
            x_sub_anomaly = x_sub_anomaly.reshape(1, height, width, 3)
            print(str(i)+"/"+str(j))
            #従来手法
            if name == "old_":
                #正常のスコア
                normal_score = model.evaluate(x_sub_normal, batch_size=1, verbose=0)
                img_normal[0, i*move:i*move+height, j*move:j*move+width, 0] +=  normal_score

                #異常のスコア
                anomaly_score = model.evaluate(x_sub_anomaly, batch_size=1, verbose=0)
                img_anomaly[0, i*move:i*move+height, j*move:j*move+width, 0] +=  anomaly_score

            #提案手法
            else:
                #正常のスコア
                mu, sigma = model.predict(x_sub_normal, batch_size=1, verbose=0)
                loss = 0
                for k in range(height):
                    for l in range(width):
                        loss += 0.5 * (x_sub_normal[0,k,l,0] - mu[0,k,l,0])**2 / sigma[0,k,l,0]
                img_normal[0, i*move:i*move+height, j*move:j*move+width, 0] +=  loss

                #異常のスコア
                mu, sigma = model.predict(x_sub_anomaly, batch_size=1, verbose=0)
                loss = 0
                for k in range(height):
                    for l in range(width):
                        loss += 0.5 * (x_sub_anomaly[0,k,l,0] - mu[0,k,l,0])**2 / sigma[0,k,l,0]
                img_anomaly[0, i*move:i*move+height, j*move:j*move+width, 0] +=  loss

    save_img(x_normal, x_anomaly, img_normal, img_anomaly, name)

def get_result_evaluate(model, x_normal, x_anomaly, name, height=16, width=16, move=2):
    img_normal = np.zeros((x_normal.shape))
    img_anomaly = np.zeros((x_normal.shape))

    for i in range(int((x_normal.shape[1]-height)/move)):
        for j in range(int((x_normal.shape[2]-width)/move)):
            x_sub_normal = x_normal[0, i*move:i*move+height, j*move:j*move+width, :]
            x_sub_anomaly = x_anomaly[0, i*move:i*move+height, j*move:j*move+width, :]
            x_sub_normal = x_sub_normal.reshape(1, height, width, 3)
            x_sub_anomaly = x_sub_anomaly.reshape(1, height, width, 3)

            #正常のスコア
            mu, sigma = model.predict(x_sub_normal, batch_size=1, verbose=0)
            loss = 0
            for k in range(height):
                for l in range(width):
                    loss += 0.5 * (x_sub_normal[0,k,l,0] - mu[0,k,l,0])**2 / sigma[0,k,l,0]
            img_normal[0, i*move:i*move+height, j*move:j*move+width, 0] +=  loss

            #異常のスコア
            mu, sigma = model.predict(x_sub_anomaly, batch_size=1, verbose=0)
            loss = 0
            for k in range(height):
                for l in range(width):
                    loss += 0.5 * (x_sub_anomaly[0,k,l,0] - mu[0,k,l,0])**2 / sigma[0,k,l,0]
            img_anomaly[0, i*move:i*move+height, j*move:j*move+width, 0] +=  loss

    img_max = np.max([img_normal, img_anomaly])
    img_min = np.min([img_normal, img_anomaly])
    img_normal = (img_normal-img_min)/(img_max-img_min) * 9 + 1
    img_anomaly = (img_anomaly-img_min)/(img_max-img_min) * 9 + 1
    return  img_anomaly[0,:,:,0],img_normal[0,:,:,0]


#16×16のサイズに切り出す
def cut_img(x, number, height=16, width=16, move = 2):
    print("cutting images ...")
    x_out = []

    x_shape = x.shape
    print(x.shape)
    for i in range(number):
        shape_0 = np.random.randint(0,x_shape[0])
        shape_1 = np.random.randint(0,x_shape[1]-height)
        shape_2 = np.random.randint(0,x_shape[2]-width)
        temp = x[shape_0, shape_1:shape_1+height, shape_2:shape_2+width, :]
        x_out.append(temp.reshape((height, width, x_shape[3])))

    print("Complete.")
    x_out = np.array(x_out)

    # for img in x:
    #     for i in range(int((x.shape[1]-height)/move + 1)):
    #         for j in range(int((x.shape[2]-width)/move +1)):
    #             x_out.append(img[i*move:i*move+height, j*move:j*move+width,:].reshape(height,width,3))
    #
    #     if len(x_out) > number:
    #         break
    # x_out = np.array(x_out)
    return x_out

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def print_eval(vae, dir_name, base_img):
    origin = cv2.imread(base_img)
    #test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    origin = origin.reshape(1,data_shape[0],data_shape[1],data_shape[2])
    origin = origin / 255

    total_anomaly = 0
    for file_name in os.listdir(dir_name):
        test = cv2.imread(dir_name + "/" + file_name)
        test = test.reshape(1,data_shape[0],data_shape[1],data_shape[2])
        test = test / 255
        anomaly_img,normal_img = get_result_evaluate(vae, origin, test, "new_", input_shape[0], input_shape[1],move)
        sub_img = anomaly_img - normal_img
        if np.amax(sub_img) > 6:
            total_anomaly += 1
        print(file_name," number of anomaly is: ",total_anomaly)

def create_model()
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Conv2D(Nc, kernel_size=2, strides=2)(inputs)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(2*Nc, kernel_size=2, strides=2)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4*Nc, kernel_size=2, strides=2)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    # exit()
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(2*2*16)(latent_inputs)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((2,2,16))(x)

    x = Conv2DTranspose(2*Nc, kernel_size=2, strides=2, padding='same')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(Nc, kernel_size=2, strides=2, padding='same')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x1 = Conv2DTranspose(3, kernel_size=4, padding='same')(x)
    # x1 = BatchNormalization()(x1)
    out1 = Activation('sigmoid')(x1)#out.shape=(n,28,28,1)

    x2 = Conv2DTranspose(3, kernel_size=4, padding='same')(x)
    # x2 = BatchNormalization()(x2)
    out2 = Activation('sigmoid')(x2)#out.shape=(n,28,28,1)

    decoder = Model(latent_inputs, [out1, out2], name='decoder')
    decoder.summary()
    #exit()
    # build VAE model
    outputs_mu, outputs_sigma_2 = decoder(encoder(inputs)[2])
    vae = Model(inputs, [outputs_mu, outputs_sigma_2], name='vae_mlp')
    #vae.summary()


    # VAE loss
    m_vae_loss = (K.flatten(inputs) - K.flatten(outputs_mu))**2 / K.flatten(outputs_sigma_2)
    m_vae_loss = 0.5 * K.sum(m_vae_loss)

    a_vae_loss = K.log(2 * 3.14 * K.flatten(outputs_sigma_2))
    a_vae_loss = 0.5 * K.sum(a_vae_loss)

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(kl_loss + m_vae_loss + a_vae_loss)
    vae.add_loss(vae_loss)
    return  vae

def train_vae(model_name, data_num, dir_train, dir_validate = None):
    vae = create_model()
    vae.compile(optimizer='adam')

    for iterator in range(1):
        # lego data
        x_train = []
        for file_name in os.listdir(dir_train):
            # image = cv2.imread(dir_train + file_name)
            print(random.choice(os.listdir(dir_train)))
            image = cv2.imread(dir_train+random.choice(os.listdir(dir_train)))
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image.reshape(data_shape[0],data_shape[1],data_shape[2])
            image = image / 255
            x_train.append(image)
            if len(x_train) > 500:
                break
        x_train = np.array(x_train)
        x_train = cut_img(x_train, data_num, input_shape[0], input_shape[1])
        print(x_train.shape)

        #lego validate data
        # x_validate = []
        # dir_validate = '../lego/microscope/OK/'
        # dir_validate = '../lego/result/OK/'
        # for file_name in os.listdir(dir_validate):
        #     image = cv2.imread(dir_validate + file_name)
        #     #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #     image = image.reshape(data_shape[0],data_shape[1],data_shape[2])
        #     image = image / 255
        #     x_validate.append(image)
        #     if len(x_validate) > 50:
        #         break
        # x_validate = np.array(x_validate)
        # x_validate = cut_img(x_validate, 10000, input_shape[0], input_shape[1])
        # print(x_validate.shape)

        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size)
                # validation_data=(x_test, None))
    vae.save_weights("model/" +  model_name+".h5")
    json_string  = vae.to_json()
    open("model/" +  model_name+ '.json', 'w').write(json_string)
def load_model(model_name):
    json_string = open("model/"+ model_name+".json").read()
    vae = model_from_json(json_string)
    vae.load_weights("model/" + model_name+".h5")
    vae.get_layer("encoder").summary()
    return  vae

def load_weights(model_name):
    vae = create_model()
    vae.load_weights("model/" + model_name+".h5")
    vae.compile(optimizer="adam")
    return  vae

if __name__=="__main__":
    # dir_train = '../lego/microscope/OK/'
    dir_train = '../lego/result/OK/'
    # train_vae("lego",1000000, dir_train)
    vae = load_model("lego")
    # vae = load_weights("best")

    #正常/異常のテストデータ
    test_normal = '../lego/result/OK/000013.jpg.jpg'
    # test_normal = cv2.imread('../lego/microscope/OK/000030.jpg')
    # test_normal = cv2.imread('../lego/microscope/OK/000110.jpg')


    # test_anomaly = cv2.imread('../lego/microscope/NG_fake/000000.jpg')
    # test_anomaly = cv2.imread('../lego/microscope/NG_fake/test.jpg')
    # test_anomaly = cv2.imread('../lego/microscope/NG/012712.jpg')
    # test_anomaly = cv2.imread('../lego/microscope/NG/012440.jpg')
    # test_anomaly = cv2.imread('../lego/microscope/NG/012779.jpg')
    # test_anomaly = cv2.imread('../lego/microscope/NG/013645.jpg')

    # 1 phan
    # test_anomaly = '../lego/result/test/OK/000459.jpg.jpg'
    # test_anomaly = '../lego/result/test/NG/011940.jpg.jpg'
    test_anomaly = '../lego/result/test/NG_fake/test000140.jpg'

    # print_eval(vae,"../lego/result/test/NG_fake",test_normal)
    # print_eval(vae,"../lego/result/OK",test_normal)
    # exit()

    #提案手法の可視化
    evaluate_img(vae, test_normal, test_anomaly, "new_", input_shape[0],input_shape[1],move)

