import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import time
import random
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors

f = h5py.File('./file.h5', 'w')
# network parameters
input_shape=(16, 16, 3)
move = 8
batch_size = 64
latent_dim = 2
epochs = 5
Nc = 16
# dataset
data_shape = [84,84,3]

class Model:
    def __init__(self, img_height=64, img_width=64, img_channel=3,num_epoch=5, batch_size=200):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.num_epoch = num_epoch
        self.batch_size = batch_size

        # session param
        self.input = None
        self.outputs_sigma_2 = None
        self.outputs_mu = None
        self.z_mean = None
        self.z_log_var = None
        self.score = None
        self.session = None

    def init_model(self, namespace='model', training=True):
        with tf.variable_scope(namespace):
            self.input = tf.placeholder(tf.float32, shape=[None, self.img_height, self.img_width, self.img_channel])
            self.out_put = self.input

            # build encoder
            for i in range(0,3):
                self.out_put = tf.layers.conv2d(
                    self.out_put,
                    filters=Nc*(2**i),
                    kernel_size=(3,3),
                    padding='same',
                    activation=tf.nn.relu
                )
                self.out_put = tf.layers.max_pooling2d(
                    self.out_put,
                    strides=(2,2),
                    pool_size=(2,2)
                )
            self.out_put = tf.layers.flatten(self.out_put)

            self.z_mean = tf.layers.dense(self.out_put,units=latent_dim)
            self.z_log_var = tf.layers.dense(self.out_put, units=latent_dim)
            self.out_put = self.sampling(self.z_mean, self.z_log_var)

            # build decoder
            self.out_put = tf.layers.dense(self.out_put,units=2*2*64,activation=tf.nn.relu)
            self.out_put = tf.reshape(self.out_put,[-1,2,2,64])
            for i in range(3,0,-1):
                self.out_put = tf.layers.conv2d_transpose(self.out_put,filters=Nc*(2*i),kernel_size=(3,3),strides=(2,2),padding='same',activation=tf.nn.relu)

            self.outputs_mu = tf.layers.conv2d_transpose(self.out_put, filters=3,kernel_size=(4,4),strides=(1,1),padding='same',activation=tf.nn.sigmoid)
            self.outputs_sigma_2 = tf.layers.conv2d_transpose(self.out_put, filters=3,kernel_size=(4,4),strides=(1,1),padding='same',activation=tf.nn.sigmoid)
            self.score = tf.reduce_sum(0.5 * (tf.reduce_mean(self.input,axis=[3]) - tf.reduce_mean(self.outputs_mu,axis=[3]))**2 / tf.reduce_mean(self.outputs_sigma_2,axis=[3]), axis=[1,2])

            cost = self.compute_loss(self.input, self.z_log_var,self.z_mean,self.outputs_mu,self.outputs_sigma_2)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(cost)
            return  self.score, cost, train_op

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def sampling(self,z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = tf.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_loss(self,inputs, z_log_var, z_mean, outputs_mu,outputs_sigma_2):
        m_vae_loss = (tf.layers.flatten(inputs) - tf.layers.flatten(outputs_mu))**2 / tf.layers.flatten(outputs_sigma_2)
        m_vae_loss = 0.5 * tf.reduce_sum(m_vae_loss)

        a_vae_loss = tf.log(2 * 3.14 * tf.layers.flatten(outputs_sigma_2))
        a_vae_loss = 0.5 * tf.reduce_sum(a_vae_loss)

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = tf.reduce_mean(kl_loss + m_vae_loss + a_vae_loss)
        return vae_loss

    #16×16のサイズに切り出す
    def cut_img(self,x, number, height=16, width=16):
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
        return x_out

    def create_data(self,train_folder,data_num):
        x_train = []
        for file_name in os.listdir(train_folder):
            # image = cv2.imread(dir_train + file_name)
            print(random.choice(os.listdir(train_folder)))
            image = cv2.imread(train_folder+random.choice(os.listdir(train_folder)))
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image.reshape(data_shape[0],data_shape[1],data_shape[2])
            image = image / 255
            x_train.append(image)
            if len(x_train) > 500:
                break
        x_train = np.array(x_train)
        x_train = self.cut_img(x_train, data_num, self.img_height, self.img_width)
        print(x_train.shape)
        return  x_train

    def train(self, train_ok_folder='./train/', model_path='./model/', model_id=0, data_num=100000, resume=False):
        tf.reset_default_graph()
        self.score , cost, train_op = self.init_model(namespace='model_' + str(model_id), training=True)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6
        self.session = tf.Session(config=config)
        if resume:
            saver.restore(self.session, model_path)
        else:
            self.session.run(tf.global_variables_initializer())

        #get data
        aver_train_loss = 0
        for i in range(self.num_epoch):
            x_train = self.create_data(train_ok_folder,data_num)
            for j in range(0, data_num, self.batch_size):
                end_j = min(data_num, j+self.batch_size)
                x_train_batch = np.float32(x_train[j:end_j])
                score, loss, _ = self.session.run([self.score,cost, train_op], feed_dict={self.input:x_train_batch})
                aver_train_loss += loss*(end_j - j)
                sys.stdout.write('\rEpoch ' +  str(i) + ' Train Progress ' + str(j) + ' Loss ' + str(loss) )

            aver_train_loss = aver_train_loss/data_num
            print('\nModel ID', model_id,'Average Loss', aver_train_loss)
        saver.save(self.session, model_path)

    def load(self,model_path='./model/',model_id=0):
        tf.reset_default_graph()
        self.score ,cost, train_op = self.init_model(namespace='model_' + str(model_id), training=True)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6
        self.session = tf.Session(config=config)
        saver.restore(self.session, model_path)

    #ヒートマップの計算
    def evaluate_img(self, x_normal_path, x_anomaly_path, height=16, width=16, move=2, im_show=False):
        start = time.time()
        x_normal = cv2.imread(x_normal_path)
        x_normal = x_normal.reshape(1,data_shape[0],data_shape[1],data_shape[2])
        x_normal = x_normal / 255

        x_anomaly = cv2.imread(x_anomaly_path)
        x_anomaly = x_anomaly.reshape(1,data_shape[0],data_shape[1],data_shape[2])
        x_anomaly = x_anomaly / 255
        img_normal = np.zeros((x_normal.shape))
        img_anomaly = np.zeros((x_anomaly.shape))
        total_time = 0
        x_sub_normal_list = []
        x_sub_anomaly_list = []
        for i in range(int((x_normal.shape[1]-height)/move)):
            for j in range(int((x_normal.shape[2]-width)/move)):
                x_sub_normal = x_normal[0, i*move:i*move+height, j*move:j*move+width, :]
                x_sub_anomaly = x_anomaly[0, i*move:i*move+height, j*move:j*move+width, :]
                x_sub_normal = x_sub_normal.reshape(height, width, 3)
                x_sub_anomaly = x_sub_anomaly.reshape( height, width, 3)
                x_sub_normal_list.append(x_sub_normal)
                x_sub_anomaly_list.append(x_sub_anomaly)
                # print(str(i)+"/"+str(j))

        #正常のスコア
        loss = self.session.run([self.score],feed_dict={self.input:np.float32(x_sub_normal_list)})

        #異常のスコア
        loss_ano = self.session.run([self.score],feed_dict={self.input:np.float32(x_sub_anomaly_list)})
        z = 0
        for i in range(int((x_normal.shape[1]-height)/move)):
            for j in range(int((x_normal.shape[2]-width)/move)):
                #正常のスコア
                img_normal[0, i*move:i*move+height, j*move:j*move+width, 0] +=  loss[0][z]

                #異常のスコア
                img_anomaly[0, i*move:i*move+height, j*move:j*move+width, 0] +=  loss_ano[0][z]
                z +=1

        if im_show:
            self.save_img(x_normal, x_anomaly, img_normal, img_anomaly)
        else:
            img_max = np.max([img_normal, img_anomaly])
            img_min = np.min([img_normal, img_anomaly])
            img_normal = (img_normal-img_min)/(img_max-img_min) * 9 + 1
            img_anomaly = (img_anomaly-img_min)/(img_max-img_min) * 9 + 1
            end = time.time()
            total_time += (end-start)
            print(str(total_time))
            return img_anomaly[0,:,:,0]-img_normal[0,:,:,0]

    def print_eval(self,dir_name, base_img):
        total_anomaly = 0
        for file_name in os.listdir(dir_name):
            test = dir_name + "/" + file_name
            result_img = self.evaluate_img( base_img, test, input_shape[0], input_shape[1],move)
            # if len(np.where(sub_img > 6)[0]) > 32:
            if np.amax(result_img) > 8:
                total_anomaly += 1
            print(file_name," number of anomaly is: ",total_anomaly)

    #ヒートマップの描画
    def save_img(self,x_normal, x_anomaly, img_normal, img_anomaly):
        path = 'images/'
        if not os.path.exists(path):
            os.mkdir(path)

        #　※注意　評価したヒートマップを1～10に正規化
        img_max = np.max([img_normal, img_anomaly])
        img_min = np.min([img_normal, img_anomaly])
        img_normal = (img_normal-img_min)/(img_max-img_min) * 9 + 1
        img_anomaly = (img_anomaly-img_min)/(img_max-img_min) * 9 + 1
        cv2.imwrite(path+'dkm.png',img_normal[0,:,:,0])
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

        plt.title("normal")

        plt.subplot(2, 2, 3)
        plt.imshow(x_anomaly[0,:,:,0], cmap='gray')
        plt.axis('off')
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.imshow(img_anomaly[0,:,:,0]-img_normal[0,:,:,0], cmap='Blues',norm=colors.LogNorm())
        plt.axis('off')
        plt.colorbar()
        plt.clim(1, 10)

        plt.title("anomaly")

        plt.savefig(path + "test.png")
        plt.show()
        plt.close()

model = Model(
    img_height=input_shape[1],
    img_width=input_shape[0],
    img_channel=input_shape[2],
    batch_size=batch_size,
    num_epoch=epochs)

temp = 5
# model.train(
#     train_ok_folder='./lines/train/0' + str(temp) + '/',
#     model_path='./model_tf/model_0' + str(temp) + '/model',
#     data_num=100000,
#     model_id=temp,
#     resume=False)

# model.load(model_path='./model_tf/model_0'+ str(temp) +'/model',
#            model_id=temp)


test_normal = './lines/train/0' + str(temp) + '/000009.jpg'
# test_normal = './lines/train/05/002228.png'
# test_anomaly = './lines/test/ng/05/05-2/002225.png'
# test_anomaly = './lines/test/ok/05/000671.png'
# test_anomaly = './lines/test.png'
total = 0
list_model = []
for temp in range(14):
    model = Model(
        img_height=input_shape[1],
        img_width=input_shape[0],
        img_channel=input_shape[2],
        batch_size=batch_size,
        num_epoch=epochs)
    model.load(model_path='./model_tf/model_%02d' % temp +'/model',
               model_id=temp)
    list_model.append(model)

total = 0
for temp in range(14):
    start = time.time()
    list_model[temp].print_eval("./test_speed/%02d" % temp +""  ,test_normal)
    total += time.time()-start
print("total time is:",total)
total = 0
for temp in range(14):
    start = time.time()
    list_model[temp].print_eval("./test_speed/%02d" % temp +""  ,test_normal)
    total += time.time()-start
print("total time is:",total)
total = 0
for temp in range(14):
    start = time.time()
    list_model[temp].print_eval("./test_speed/%02d" % temp +""  ,test_normal)
    total += time.time()-start
print("total time is:",total)



# model.print_eval("./lines/test/ok/0" + str(temp) +"_backup"  ,test_normal)
# print("*****************************")
# model.print_eval("./lines/test/ng/05/05-1_backup",test_normal)
# print("*****************************")
# model.print_eval("./lines/test/ng/05/05-2_backup",test_normal)
# exit()
#model.evaluate_img(test_normal, test_anomaly, input_shape[0],input_shape[1],move, im_show=True)