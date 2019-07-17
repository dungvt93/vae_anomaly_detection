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
batch_size = 32
latent_dim = 2
epochs = 5
Nc = 16
# dataset
data_shape = [64,64,3]

class Model:
    def __init__(self):
        #variable for model
        self.number_of_model = 0
        self.session = None
        self.input_list = []
        self.score_list = []
        self.cost_list = []
        self.train_op_list = []
        self.outputs_mu_list = []

    @staticmethod
    def init_model(namespace='model', input=None, training=True):
        with tf.variable_scope(namespace):
            out_put  = input

            # build encoder
            for i in range(0,3):
                out_put = tf.layers.conv2d(
                    out_put,
                    filters=Nc*(2**i),
                    kernel_size=(3,3),
                    padding='same',
                    activation=tf.nn.relu
                )
                out_put = tf.layers.max_pooling2d(
                    out_put,
                    strides=(2,2),
                    pool_size=(2,2)
                )
            out_put = tf.layers.flatten(out_put)
            z_mean = tf.layers.dense(out_put,units=latent_dim)
            z_log_var = tf.layers.dense(out_put, units=latent_dim)
            out_put = Model.sampling(z_mean, z_log_var)

            # build decoder
            out_put = tf.layers.dense(out_put,units=2*2*64,activation=tf.nn.relu)
            out_put = tf.reshape(out_put,[-1,2,2,64])
            for i in range(3,0,-1):
                out_put = tf.layers.conv2d_transpose(out_put,filters=Nc*(2*i),kernel_size=(3,3),strides=(2,2),padding='same',activation=tf.nn.relu)

            outputs_mu = tf.layers.conv2d_transpose(out_put, filters=3,kernel_size=(4,4),strides=(1,1),padding='same',activation=tf.nn.sigmoid)
            outputs_sigma_2 = tf.layers.conv2d_transpose(out_put, filters=3,kernel_size=(4,4),strides=(1,1),padding='same',activation=tf.nn.sigmoid)
            score = tf.reduce_sum(0.5 * (tf.reduce_mean(input,axis=[3]) - tf.reduce_mean(outputs_mu,axis=[3]))**2 / tf.reduce_mean(outputs_sigma_2,axis=[3]), axis=[1,2])

            cost = Model.compute_loss(input, z_log_var,z_mean,outputs_mu,outputs_sigma_2)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(cost)
            return  score, cost, train_op, outputs_mu

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    @staticmethod
    def sampling(z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = tf.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    @staticmethod
    def compute_loss(inputs, z_log_var, z_mean, outputs_mu,outputs_sigma_2):
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
    @staticmethod
    def cut_img(x, number, height=16, width=16):
        print("cutting images ...")
        x_out = []

        x_shape = x.shape
        print(x.shape)
        for i in range(number):
            shape_0 = np.random.randint(0,x_shape[0])
            shape_1 = np.random.randint(0,x_shape[1]-height)
            shape_2 = np.random.randint(0,x_shape[2]-width)
            temp_img = x[shape_0, shape_1:shape_1+height, shape_2:shape_2+width, :]
            x_out.append(temp_img.reshape((height, width, x_shape[3])))

        print("Complete.")
        x_out = np.array(x_out)
        return x_out

    @staticmethod
    def create_data(train_folder,data_num):
        x_train = []
        for file_name in os.listdir(train_folder):
            image = cv2.imread(train_folder + file_name)
            # print(file_name)
            # print(random.choice(os.listdir(train_folder)))
            # image = cv2.imread(train_folder+random.choice(os.listdir(train_folder)))

            image = image.reshape(data_shape[0],data_shape[1],data_shape[2])
            image = image / 255
            x_train.append(image)
            # if len(x_train) > 24:
            #     break
        x_train = np.array(x_train)
        x_train = Model.cut_img(x_train, data_num, input_shape[0], input_shape[1])
        print(x_train.shape)
        return  x_train

    # param:
    # 1. folder image for trainning
    # 2. folder to save model
    # 3. number of model in combine model
    # 4. number of data for trainning (cut from image in folder image)
    # 5. sequence of model in combine model will be used to trainning
    # 6. True if create new combine model, False if load exist combine model
    # 7. True if continue trainning single model in combine model, False if reset and trainning single model in combine model
    def train(self, train_folder='./train/', model_path='./model/',number_of_model = 9, model_id=0, data_num=100000, new_combine_model=False, resume_single_model=False):
        self.number_of_model = number_of_model
        tf.reset_default_graph()
        for i in range(number_of_model):
            self.input_list.append(tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]]))
            score , cost, train_op, _ = Model.init_model(namespace='model_' + str(i),input=self.input_list[i], training=True)
            self.score_list.append(score)
            self.cost_list.append(cost)
            self.train_op_list.append(train_op)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6
        self.session = tf.Session(config=config)
        if new_combine_model:
            self.session.run(tf.global_variables_initializer())
        else:
            saver.restore(self.session, model_path)
            if not resume_single_model:
                self.session.run(tf.initialize_variables(tf.global_variables('model_' + str(model_id) + '\D')))

        #get data
        aver_train_loss = 0
        for i in range(epochs):
            x_train = Model.create_data(train_folder,data_num)
            for j in range(0, data_num, batch_size):
                end_j = min(data_num, j+batch_size)
                x_train_batch = np.float32(x_train[j:end_j])
                score, loss, _ = self.session.run([self.score_list[model_id],self.cost_list[model_id], self.train_op_list[model_id]], feed_dict={self.input_list[model_id]:x_train_batch})
                aver_train_loss += loss*(end_j - j)
                sys.stdout.write('\rEpoch ' +  str(i) + ' Train Progress ' + str(j) + ' Loss ' + str(loss) )

            aver_train_loss = aver_train_loss/data_num
            print('\nModel ID', model_id,'Average Loss', aver_train_loss)
        saver.save(self.session, model_path)

    def load_model(self,model_path='./model/', number_of_model=9):
        self.number_of_model = number_of_model
        tf.reset_default_graph()
        for i in range(number_of_model):
            self.input_list.append(tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]]))
            score , cost, train_op, output_mu, = Model.init_model(namespace='model_' + str(i), input=self.input_list[i], training=True)
            self.score_list.append(score)
            self.cost_list.append(cost)
            self.train_op_list.append(train_op)
            self.outputs_mu_list.append(output_mu)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6
        self.session = tf.Session(config=config)
        saver.restore(self.session, model_path)

    @staticmethod
    def find_compare_img(x_anomaly, model_id):
        mean_img = np.mean(x_anomaly,axis = 2)
        index_start = np.where(mean_img > 40)[1][0]
        return cv2.imread("./%02d_compare/%05d.png" % (model_id,index_start))

    #ヒートマップの計算
    def evaluate_img(self, x_normal_path, x_anomaly_path, model_id=0, im_show=False):
        height = input_shape[0]
        width = input_shape[1]

        x_anomaly = cv2.imread(x_anomaly_path)
        #find compare img
        if x_normal_path is None:
            x_normal = Model.find_compare_img(x_anomaly,model_id)
            # print("compare image:" ,x_normal_path)
        else:
            x_normal = cv2.imread(x_normal_path)

        x_anomaly = x_anomaly.reshape(data_shape[0],data_shape[1],data_shape[2])
        x_anomaly = x_anomaly / 255
        x_normal = x_normal.reshape(data_shape[0],data_shape[1],data_shape[2])
        x_normal = x_normal / 255

        img_normal = np.zeros((x_normal.shape))
        img_anomaly = np.zeros((x_anomaly.shape))
        x_sub_normal_list = []
        x_sub_anomaly_list = []
        for i in range(int((x_normal.shape[0]-height)/move)+1):
            for j in range(int((x_normal.shape[1]-width)/move)+1):
                x_sub_normal = x_normal[i*move:i*move+height, j*move:j*move+width, :]
                x_sub_anomaly = x_anomaly[i*move:i*move+height, j*move:j*move+width, :]
                x_sub_normal = x_sub_normal.reshape(height, width, 3)
                x_sub_anomaly = x_sub_anomaly.reshape(height, width, 3)
                x_sub_normal_list.append(x_sub_normal)
                x_sub_anomaly_list.append(x_sub_anomaly)
                # print(str(i)+"/"+str(j))

        #正常のスコア
        loss = self.session.run([self.score_list[model_id]],feed_dict={self.input_list[model_id]:np.float32(x_sub_normal_list)})

        #異常のスコア
        loss_ano = self.session.run([self.score_list[model_id]],feed_dict={self.input_list[model_id]:np.float32(x_sub_anomaly_list)})

        if im_show:
            z = 0
            for i in range(int((x_normal.shape[0]-height)/move)+1):
                for j in range(int((x_normal.shape[1]-width)/move)+1):
                    #正常のスコア
                    img_normal[i*move:i*move+height, j*move:j*move+width, 0] +=  loss[0][z]

                    #異常のスコア
                    img_anomaly[i*move:i*move+height, j*move:j*move+width, 0] +=  loss_ano[0][z]
                    z +=1

            img_normal, img_anomaly = Model.recalculate_conner_score(img_normal,img_anomaly)
            self.save_loss_map(img_normal,img_anomaly)
            self.save_heat_map(x_normal, x_anomaly, img_normal, img_anomaly)
        else:
            return Model.is_anomaly(loss[0],loss_ano[0],height,width)

    @staticmethod
    def save_loss_map(img_normal,img_anomaly):
        if not os.path.exists("loss_file"):
            os.mkdir("loss_file")
        # with open("loss_file/img_normal","wb") as fp:
        #     pickle.dump(img_normal,fp)
        # with open("loss_file/img_anomaly","wb") as fp:
        #     pickle.dump(img_anomaly,fp)
        img_normal = img_normal[:,:,0]
        img_anomaly = img_anomaly[:,:,0]
        img_normal = np.reshape(img_normal, [8,8,8,8])
        img_normal = img_normal[:,0,:,0]
        img_normal = np.reshape(img_normal, [8,8])

        img_anomaly = np.reshape(img_anomaly, [8,8,8,8])
        img_anomaly = img_anomaly[:,0,:,0]
        img_anomaly = np.reshape(img_anomaly, [8,8])
        # np.savetxt("loss_file/img_normal",img_normal,'%5d')
        np.savetxt("loss_file/img_anomaly",img_anomaly,'%5d')

    #ヒートマップの描画
    @staticmethod
    def save_heat_map(x_normal, x_anomaly, img_normal, img_anomaly):
        path = 'images/'
        if not os.path.exists(path):
            os.mkdir(path)

        #　※注意　評価したヒートマップを1～10に正規化
        img_max = np.max([img_normal[:,:,0], img_anomaly[:,:,0]])
        img_min = np.min([img_normal[:,:,0], img_anomaly[:,:,0]])
        img_normal = (img_normal[:,:,0]-img_min)/(img_max-img_min) * 9 + 1
        img_anomaly = (img_anomaly[:,:,0]-img_min)/(img_max-img_min) * 9 + 1

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(x_normal[:,:,0], cmap='gray')
        plt.axis('off')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.imshow(img_normal[:,:], cmap='Blues',norm=colors.LogNorm())
        plt.axis('off')
        plt.colorbar()
        plt.clim(1, 10)

        plt.title("normal")

        plt.subplot(2, 2, 3)
        plt.imshow(x_anomaly[:,:,0], cmap='gray')
        plt.axis('off')
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.imshow(img_anomaly[:,:]-img_normal[:,:], cmap='Blues',norm=colors.LogNorm())
        plt.axis('off')
        plt.colorbar()
        plt.clim(1, 10)

        plt.title("anomaly")

        plt.savefig(path + "test.png")
        plt.show()
        plt.close()

    # check single image is anomaly or not
    # param:
    # 1. list of loss of compare image
    # 2. list of loss of test image
    # 3. height of input model
    # 4. width of input model
    # output: True if anomaly, False if normal
    @staticmethod
    def is_anomaly(loss,loss_ano,height,width, thread_hold=7.5, menseki=1):
        result = False
        img_normal = np.zeros(data_shape)
        img_anomaly = np.zeros(data_shape)
        z = 0
        for i in range(int((data_shape[0]-height)/move)+1):
            for j in range(int((data_shape[1]-width)/move)+1):

                #正常のスコア
                img_normal[i*move:i*move+height, j*move:j*move+width, 0] +=  loss[z]

                #異常のスコア
                img_anomaly[i*move:i*move+height, j*move:j*move+width, 0] +=  loss_ano[z]
                z +=1

        img_normal, img_anomaly = Model.recalculate_conner_score(img_normal,img_anomaly)
        # only use reconstruct loss
        # if np.amax(img_anomaly > 2000):
        #     result = True

        img_max = np.max([img_normal[:,:,0], img_anomaly[:,:,0]])
        img_min = np.min([img_normal[:,:,0], img_anomaly[:,:,0]])
        img_normal = (img_normal[:,:,0]-img_min)/(img_max-img_min) * 9 + 1
        img_anomaly = (img_anomaly[:,:,0]-img_min)/(img_max-img_min) * 9 + 1

        img_result = img_anomaly[:,:]-img_normal[:,:]

        img_result = np.reshape(img_result, [int(data_shape[0]/move),move,int(data_shape[1]/move),move])
        img_result = img_result[:,0,:,0]
        img_result = np.reshape(img_result, [int(data_shape[0]/move),int(data_shape[1]/move)])

        # print(np.amax(img_result))
        if menseki == 1:
            if np.amax(img_result) > thread_hold:
                result = True
        else:
            point_list = list(np.where(img_result > thread_hold))
            number_cell_ano = len(point_list[0])
            #check area of anomaly
            if number_cell_ano >= menseki:
                while len(point_list[0]) > 0:
                    # pop first element
                    x = point_list[0][0]
                    point_list[0] = np.delete(point_list[0],0)
                    y = point_list[1][0]
                    point_list[1] = np.delete(point_list[1],0)
                    if Model.max_area_of_point(list((point_list[0],point_list[1])),x,y,1) >= menseki:
                        result = True
                        break
        return result

    #get max area of point (number of point adjacent)
    @staticmethod
    def max_area_of_point(point_list, point_x,point_y,area=1):
        result = [area]
        temp_list = [[],[]]
        while len(point_list[0]) > 0:
            x = point_list[0][0]
            y = point_list[1][0]
            if abs(x-point_x) <= 1 and abs(y-point_y) <= 1:
                # remove point have been checked
                point_list[0] = np.delete(point_list[0],0)
                point_list[1] = np.delete(point_list[1],0)
                result.append(Model.max_area_of_point(list((point_list[0],point_list[1])),x,y,area+1))
            else:
                # save point have'nt been checked'
                temp_list[0].append(x)
                temp_list[1].append(y)
                point_list[0] = np.delete(point_list[0],0)
                point_list[1] = np.delete(point_list[1],0)
        point_list[0] = temp_list[0]
        point_list[1] = temp_list[1]
        return max(result)


    @staticmethod
    def recalculate_conner_score(img_normal, img_anomaly):
        #recalculate conner cell loss
        for i in (0,data_shape[0]-move):
            img_normal[i:i+move,:,0] = img_normal[i:i+move,:,0]* 2
            img_anomaly[i:i+move,:,0] = img_anomaly[i:i+move,:,0]*2
            img_normal[:,i:i+move,0] = img_normal[:,i:i+move,0]*2
            img_anomaly[:,i:i+move,0] = img_anomaly[:,i:i+move,0]*2
        return img_normal, img_anomaly

    @staticmethod
    def get_max_loss_level(loss,loss_ano,height,width):
        img_normal = np.zeros(data_shape)
        img_anomaly = np.zeros(data_shape)
        z = 0
        for i in range(int((data_shape[0]-height)/move)+1):
            for j in range(int((data_shape[1]-width)/move)+1):
                #正常のスコア
                img_normal[i*move:i*move+height, j*move:j*move+width, 0] +=  loss[z]

                #異常のスコア
                img_anomaly[i*move:i*move+height, j*move:j*move+width, 0] +=  loss_ano[z]
                z +=1

        img_normal, img_anomaly = Model.recalculate_conner_score(img_normal,img_anomaly)

        img_max = np.max([img_normal[:,:,0], img_anomaly[:,:,0]])
        img_min = np.min([img_normal[:,:,0], img_anomaly[:,:,0]])
        img_normal = (img_normal[:,:,0]-img_min)/(img_max-img_min) * 9 + 1
        img_anomaly = (img_anomaly[:,:,0]-img_min)/(img_max-img_min) * 9 + 1

        img_result = img_anomaly[:,:]-img_normal[:,:]

        return np.amax(img_result)

    def print_eval(self,dir_name, base_img, model_id):
        total_anomaly = 0
        # dump code
        # error_list = []
        for file_name in os.listdir(dir_name):
            test = dir_name + "/" + file_name
            result = self.evaluate_img( base_img, test, model_id=model_id)
            # if len(np.where(sub_img > 6)[0]) > 32:
            if result:
                total_anomaly += 1
                # error_list.append(file_name)
            #     print(file_name+ " NG")
            # else:
            #     print(file_name)
            print(file_name," number of anomaly is: ",total_anomaly)
        # import pickle
        # with open("dump2.txt","wb") as fp:
        #     pickle.dump(error_list,fp)

    def pre_process(self,pin_normal,list_pin_anomaly):
        test_normal_list = []
        test_anomaly_list = []
        if pin_normal is not None:
            pin_normal = cv2.resize(pin_normal,(data_shape[0],9*data_shape[1]))
        for img_1pin_anomaly in list_pin_anomaly:
            img_1pin_anomaly = cv2.resize(img_1pin_anomaly,(data_shape[0],9*data_shape[1]))
            for i in range(self.number_of_model):
                test_anomaly_list.append(img_1pin_anomaly[i*64:i*64+data_shape[0],:])
                if pin_normal is not None:
                    test_normal_list.append(pin_normal[i*64:i*64+data_shape[0],:])
                else:
                    test_normal_list.append(Model.find_compare_img(img_1pin_anomaly[i*64:i*64+data_shape[0],:],i))
        # test_normal_list = []
        # test_normal_list.append(cv2.imread('./kin_train/00/000001.png'))
        # test_normal_list.append(cv2.imread('./kin_train/01/000001.png'))
        # test_normal_list.append(cv2.imread('./kin_train/02/000001.png'))
        # test_normal_list.append(cv2.imread('./kin_train/03/000283.png'))
        # test_normal_list.append(cv2.imread('./kin_train/04/000001.png'))
        # test_normal_list.append(cv2.imread('./kin_train/05/000001.png'))
        # test_normal_list.append(cv2.imread('./kin_train/06/000237.png'))
        # test_normal_list.append(cv2.imread('./kin_train/07/000001.png'))
        # test_normal_list.append(cv2.imread('./kin_train/08/000073.png'))
        return  test_normal_list, test_anomaly_list

    # detect anomaly of real image
    # param;
    # 1. pin image for compare
    # 2. list pin image for test
    # 3. pixel move
    # output: sequence of part anomaly. None if normal image
    def detect(self, pin_normal, list_pin_anomaly):
        num_of_pin = len(list_pin_anomaly)
        x_normal_list, x_anomaly_list = self.pre_process(pin_normal, list_pin_anomaly)
        height = input_shape[0]
        width = input_shape[1]
        number_of_input = (int((data_shape[0]-height)/move) + 1)**2  #number of input_shape of each part image
        feed_dict_normal = {}
        feed_dict_anomaly = {}
        #list cut_image of 1 pin (compare image)
        x_sub_normal_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        #list cut_image of 1 pin (test image)
        x_sub_anomaly_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        # preprocess data
        for index, (x_normal, x_anomaly) in enumerate(zip(x_normal_list,x_anomaly_list)):
            x_normal = x_normal.reshape(1,data_shape[0],data_shape[1],data_shape[2])
            x_normal = x_normal / 255
            x_anomaly = x_anomaly.reshape(1,data_shape[0],data_shape[1],data_shape[2])
            x_anomaly = x_anomaly / 255

            for i in range(int((x_normal.shape[1]-height)/move)+1):
                for j in range(int((x_normal.shape[2]-width)/move)+1):
                    x_sub_normal = x_normal[0, i*move:i*move+height, j*move:j*move+width, :]
                    x_sub_anomaly = x_anomaly[0, i*move:i*move+height, j*move:j*move+width, :]
                    x_sub_normal = x_sub_normal.reshape(height, width, 3)
                    x_sub_anomaly = x_sub_anomaly.reshape( height, width, 3)
                    x_sub_normal_list[index % self.number_of_model].append(x_sub_normal)
                    x_sub_anomaly_list[index % self.number_of_model].append(x_sub_anomaly)

        for m in range(self.number_of_model):
            feed_dict_normal[self.input_list[m]]  = np.float32(x_sub_normal_list[m])
            feed_dict_anomaly[self.input_list[m]] = np.float32(x_sub_anomaly_list[m])

        start = time.time()
        #正常のスコア
        loss_list = self.session.run([self.score_list],feed_dict=feed_dict_normal)

        #異常のスコア
        loss_ano_list = self.session.run([self.score_list],feed_dict=feed_dict_anomaly)
        print("time for inference",time.time() - start)
        result = []

        start = time.time()
        loss_list = np.array(loss_list[0])
        loss_ano_list = np.array(loss_ano_list[0])

        for index, (loss, loss_ano) in enumerate(zip(loss_list, loss_ano_list)): #number_of_model loop
            for i in range(int(len(loss)/number_of_input)): #number_of_real image loop
                loss_one_img = loss[i*number_of_input:(i+1)*number_of_input]
                loss_ano_one_img = loss_ano[i*number_of_input:(i+1)*number_of_input]
                result.append(Model.get_max_loss_level(loss_one_img,loss_ano_one_img,height,width))
        print("time for hitmap",time.time() - start)
        return [result[i:num_of_pin*self.number_of_model:num_of_pin] for i in range(num_of_pin)]

    def save_reconstruct_img(self,direct, model_id):
        for file_name in os.listdir(direct):
            print(file_name)
            img = cv2.imread(direct + "/" + file_name)
            img = img / 255
            food = []
            for i in range((data_shape[0]-input_shape[0])/move +1):
                for j in range((data_shape[1]-input_shape[1])/move +1):
                    food.append(img[i*move:i*move+input_shape[1], j*move:j*move+input_shape[0], :])
            result = self.session.run([self.outputs_mu_list[model_id]],feed_dict={self.input_list[model_id]:np.float32(food)})
            reconstruct = np.zeros([data_shape[0],data_shape[1],3])
            z = 0
            for i in range((data_shape[0]-input_shape[0])/move +1):
                for j in range((data_shape[1]-input_shape[1])/move +1):
                    reconstruct[i*move:i*move+input_shape[1], j*move:j*move+input_shape[0], :] = result[0][z]
                    z+=1

            if not os.path.exists("fukugen"):
                os.makedirs("fukugen")
            cv2.imwrite("fukugen/" + file_name ,reconstruct*255)

model = Model()

temp = 3

# model.train(
#    train_folder='./kin_train/%02d' % temp + '/',
#    # train_folder='./combine_train' + '/',
#    model_path='./model_kin_tf/model',
#    data_num=100000,
#    number_of_model=9,
#    model_id=temp,
#    new_combine_model=True,
#    resume_single_model=False)

model.load_model(model_path='./model_kin_tf/model',number_of_model=9)

# run 2 pin
for i in range(3):
    start = time.time()
    input_normal = cv2.imread('pin_normal.png')
    list_input_anomaly = [cv2.imread('pin_anomaly.png'),cv2.imread('pin_anomaly_3.png')]
    print(model.detect(input_normal,list_input_anomaly))
    print(time.time()-start)

test_normal = None
# test_normal = './kin_train/%02d' % temp + '_izzi/000135.png'      #03_izzi
# test_normal = './kin_train/%02d' % temp + '/000283.png'     #03
# test_normal = './kin_train/%02d' % temp + '/000237.png'    #06
# test_normal = './kin_train/%02d' % temp + '/000073.png'   #08
# test_normal = './kin_train/%02d' % temp + '/000001.png'

# test_anomaly = './kin_test/ok/%02d' % temp + '/000092.png'
# test_anomaly = 'dump/test.png'

#plus_03
# test_anomaly = './kin_test/ng/%02d' % temp + '_plus/test.png'
# model.print_eval("./kin_test/ok/%02d" % temp +"_izzi",test_normal, model_id=temp)
# print("*****************************")
# model.print_eval("./kin_test/ng/%02d" % temp +"_plus",test_normal, model_id=temp)
# print("*****************************")
# model.print_eval("./kin_test/ng/%02d" % temp +"_izzi_fail",test_normal, model_id=temp)

# model.print_eval("./kin_test/ok/%02d" % temp +"",test_normal, model_id=temp)
# print("*****************************")
# model.print_eval("./kin_test/ng/%02d" % temp,test_normal,model_id=temp)
# print("*****************************")
# model.print_eval("dump",test_normal,model_id=temp)
# exit()

# model.evaluate_img(test_normal, test_anomaly, im_show=True, model_id=temp)

# model.save_reconstruct_img("dump" ,temp)
