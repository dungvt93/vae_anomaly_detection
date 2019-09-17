#coding:utf-8
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import time
import random
import h5py
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv
import math

class Model:
	def __init__(self, compare_img_path=None, input_shape=(16, 16, 3), cell_shape = (64,64,3),img_shape=(64*9,64,3), move = 8):
		#variable for model
		self.number_of_model = 0
		self.session = None
		self.input_list = []
		self.score_list = []
		self.cost_list = []
		self.train_op_list = []
		self.outputs_mu_list = []
		self.input_shape = input_shape
		self.cell_shape = cell_shape
		self.img_shape = img_shape
		self.move = move
		self.compare_img = cv2.imread(compare_img_path)

	@staticmethod
	def init_model(namespace='model', input=None, Nc = 16, latent_dim = 6, learning_rate=0.0001):
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
			outputs_sigma_2 = outputs_sigma_2 + 1e-10
			# outputs_sigma_2 = tf.where(tf.less(outputs_sigma_2,0.000001),outputs_sigma_2+0.000001,outputs_sigma_2)
			score = tf.reduce_sum(0.5 * (tf.reduce_mean(input,axis=[3]) - tf.reduce_mean(outputs_mu,axis=[3]))**2 / tf.reduce_mean(outputs_sigma_2,axis=[3]), axis=[1,2])
			# score = outputs_sigma_2

			cost = Model.compute_loss(input, z_log_var,z_mean,outputs_mu,outputs_sigma_2)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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

		# test = -tf.math.sigmoid(tf.layers.flatten(outputs_sigma_2))
		# test = tf.reduce_sum(test)
		# exit()

		vae_loss = tf.reduce_mean(4*kl_loss + m_vae_loss +  a_vae_loss)

		# vae_loss = tf.reduce_mean(4*kl_loss + m_vae_loss +  a_vae_loss)
		# vae_loss = tf.reduce_mean(kl_loss + m_vae_loss)
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

	def create_data(self,train_folder,data_num, offset):
		x_train = []
		for idx,file_name in enumerate(os.listdir(train_folder)):
			if idx >= offset:
				if os.path.splitext(file_name)[1] != ".png" and os.path.splitext(file_name)[1] != ".jpg" :
					continue
				image = cv2.imread(train_folder + file_name)
				image = cv2.resize(image,(self.cell_shape[1],self.cell_shape[0]))
				# print(file_name)
				# print(random.choice(os.listdir(train_folder)))
				# image = cv2.imread(train_folder+random.choice(os.listdir(train_folder)))

				image = image.reshape(self.cell_shape[0],self.cell_shape[1],self.cell_shape[2])
				image = image / 255
				x_train.append(image)
			if len(x_train) >= 4000:
				break
		x_train = np.array(x_train)
		x_train = Model.cut_img(x_train, data_num, self.input_shape[0], self.input_shape[1])

		print(x_train.shape)
		return  x_train

	# param:
	# 1. folder image for trainning
	# 2. folder to save model
	# 3. number of model in combine model
	# 4. sequence of model in combine model will be used to trainning
	# 5. number of data for trainning (cut from image in folder image)
	# 6. number of batch size
	# 7. number of epochs
	# 8. True if create new combine model, False if load exist combine model
	# 9. True if continue trainning single model in combine model, False if reset and retrainning single model in combine model
	def train(self, train_folder='./train/', model_path='./model/',number_of_model = 9, model_id=0, data_num=100000, batch_size = 32, epochs = 5, new_combine_model=False, resume_single_model=False):
		self.number_of_model = number_of_model
		tf.reset_default_graph()
		for i in range(number_of_model):
			self.input_list.append(tf.placeholder(tf.float32, shape=[None, self.input_shape[0], self.input_shape[1], self.input_shape[2]]))
			score , cost, train_op, output_mu = Model.init_model(namespace='model_' + str(i),input=self.input_list[i])
			self.score_list.append(score)
			self.cost_list.append(cost)
			self.train_op_list.append(train_op)
			self.outputs_mu_list.append(output_mu)

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
		number_train_files = len(os.listdir(train_folder))
		offset = 0
		while offset < number_train_files:
			x_train = self.create_data(train_folder,data_num, offset)
			offset += 4000
			for i in range(epochs):
				aver_train_loss = 0
				for j in range(0, data_num, batch_size):
					end_j = min(data_num, j+batch_size)
					x_train_batch = np.float32(x_train[j:end_j])
					score, loss, _, output_mu = self.session.run([self.score_list[model_id],self.cost_list[model_id], self.train_op_list[model_id], self.outputs_mu_list[model_id]], feed_dict={self.input_list[model_id]:x_train_batch})

					# print(np.sum(output_mu))
					# if len(np.where(output_mu == 0)[0]) > 0:
					# 	print(np.where(output_mu == 0))
					# 	exit()
					# if np.isnan(loss):
					# if loss < -60000:
					# 	with open('dkm2.txt','a') as outfile:
					# 		for dkm in output_mu:
					# 			for dkm2 in dkm:
					# 				np.savetxt(outfile,dkm2,"%5f")
					# 	print("loss")
					# 	exit()
					aver_train_loss += loss*(end_j - j)
					sys.stdout.write('\rEpoch ' +  str(i) + ' Train Progress ' + str(j) + ' Loss ' + str(loss))
				aver_train_loss = aver_train_loss/data_num
				print('\nModel ID', model_id,'Average Loss', aver_train_loss)
				print("train status "+str(offset)+"/"+str(number_train_files))

		saver.save(self.session, model_path)

	def load_model(self,model_path='./model/', number_of_model=9):
		self.number_of_model = number_of_model
		tf.reset_default_graph()
		for i in range(number_of_model):
			self.input_list.append(tf.placeholder(tf.float32, shape=[None, self.input_shape[0], self.input_shape[1], self.input_shape[2]]))
			score , cost, train_op, output_mu, = Model.init_model(namespace='model_' + str(i), input=self.input_list[i])
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
	def evaluate_img(self, x_normal_path, x_anomaly_path, model_id=0, im_show=False,thres_hold=0):
		height = self.input_shape[0]
		width = self.input_shape[1]

		x_anomaly = cv2.imread(x_anomaly_path)
		#find compare img
		if x_normal_path is None:
			x_normal = Model.find_compare_img(x_anomaly,model_id)
			# print("compare image:" ,x_normal_path)
		else:
			x_normal = cv2.imread(x_normal_path)

		x_anomaly = cv2.resize(x_anomaly,(self.cell_shape[1],self.cell_shape[0]))
		x_anomaly = x_anomaly.reshape(self.cell_shape[0],self.cell_shape[1],self.cell_shape[2])
		x_anomaly = x_anomaly / 255
		x_normal = cv2.resize(x_normal,(self.cell_shape[1],self.cell_shape[0]))
		x_normal = x_normal.reshape(self.cell_shape[0],self.cell_shape[1],self.cell_shape[2])
		x_normal = x_normal / 255

		x_sub_normal_list = []
		x_sub_anomaly_list = []
		for i in range(int((x_normal.shape[0]-height)/self.move)+1):
			for j in range(int((x_normal.shape[1]-width)/self.move)+1):
				x_sub_normal = x_normal[i*self.move:i*self.move+height, j*self.move:j*self.move+width, :]
				x_sub_anomaly = x_anomaly[i*self.move:i*self.move+height, j*self.move:j*self.move+width, :]
				x_sub_normal = x_sub_normal.reshape(height, width, 3)
				x_sub_anomaly = x_sub_anomaly.reshape(height, width, 3)
				x_sub_normal_list.append(x_sub_normal)
				x_sub_anomaly_list.append(x_sub_anomaly)
				# print(str(i)+"/"+str(j))

		#正常のスコア
		loss = self.session.run([self.score_list[model_id]],feed_dict={self.input_list[model_id]:np.float32(x_sub_normal_list)})

		#異常のスコア
		loss_ano = self.session.run([self.score_list[model_id]],feed_dict={self.input_list[model_id]:np.float32(x_sub_anomaly_list)})

		img_normal, img_anomaly = self.create_loss_img(loss[0],loss_ano[0])
		if im_show:
			self.save_loss_map(img_normal,img_anomaly)
			self.save_heat_map(x_normal, x_anomaly, img_normal, img_anomaly)
		else:
			return self.is_anomaly(img_normal, img_anomaly, thres_hold=thres_hold)

	def create_loss_img(self,loss, loss_ano):
		img_normal = np.zeros(self.cell_shape)
		img_anomaly = np.zeros(self.cell_shape)
		z = 0
		for i in range(int((self.cell_shape[0] - self.input_shape[0]) / self.move) + 1):
			for j in range(int((self.cell_shape[1] - self.input_shape[1]) / self.move) + 1):
				# 正常のスコア
				img_normal[i * self.move:i * self.move + self.input_shape[0], j * self.move:j * self.move + self.input_shape[1], 0] += loss[z]

				# 異常のスコア
				img_anomaly[i * self.move:i * self.move + self.input_shape[0], j * self.move:j * self.move + self.input_shape[1], 0] += loss_ano[z]
				z += 1
		# img_normal, img_anomaly = self.recalculate_conner_score(img_normal,img_anomaly)
		return img_normal, img_anomaly

	def save_loss_map(self,img_normal,img_anomaly):
		if not os.path.exists("loss_file"):
			os.mkdir("loss_file")
		# with open("loss_file/img_normal","wb") as fp:
		#     pickle.dump(img_normal,fp)
		# with open("loss_file/img_anomaly","wb") as fp:
		#     pickle.dump(img_anomaly,fp)
		img_normal = img_normal[:,:,0]
		img_anomaly = img_anomaly[:,:,0]

		img_normal = np.reshape(img_normal, [int(self.cell_shape[0]/self.move),self.move,int(self.cell_shape[1]/self.move),self.move])
		img_normal = img_normal[:,0,:,0]
		img_normal = np.reshape(img_normal, [int(self.cell_shape[0]/self.move),int(self.cell_shape[1]/self.move)])


		img_anomaly = np.reshape(img_anomaly, [int(self.cell_shape[0]/self.move),self.move,int(self.cell_shape[1]/self.move),self.move])
		img_anomaly = img_anomaly[:,0,:,0]
		img_anomaly = np.reshape(img_anomaly, [int(self.cell_shape[0]/self.move),int(self.cell_shape[1]/self.move)])

		np.savetxt("loss_file/img_normal",img_normal,'%5f')
		np.savetxt("loss_file/img_anomaly",img_anomaly,'%5f')

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

	def is_anomaly(self, img_normal, img_anomaly, thres_hold=0, menseki=1):
		result = False
		# only use reconstruct loss
		# img_result = img_anomaly - img_normal
		# if np.max(img_result > 3):
		# 	return True

		img_result = self.get_subtraction_score_result(img_normal,img_anomaly)
		if menseki == 1:
			if np.amax(img_result) > thres_hold:
				result = True
		else:
			#([x1,x2,x3...],[y1,y2,y3...])
			point_list = list(np.where(img_result > thres_hold))
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

	def recalculate_conner_score(self,img_normal, img_anomaly):
		#recalculate conner cell loss
		for i in (0,self.cell_shape[0]-self.move):
			img_normal[i:i+self.move,:,0] = img_normal[i:i+self.move,:,0]* 2
			img_anomaly[i:i+self.move,:,0] = img_anomaly[i:i+self.move,:,0]*2
			img_normal[:,i:i+self.move,0] = img_normal[:,i:i+self.move,0]*2
			img_anomaly[:,i:i+self.move,0] = img_anomaly[:,i:i+self.move,0]*2
		return img_normal, img_anomaly

	def get_subtraction_score_result(self,img_normal,img_anomaly):
		img_max = np.max([img_normal[:,:,0], img_anomaly[:,:,0]])
		img_min = np.min([img_normal[:,:,0], img_anomaly[:,:,0]])
		img_normal = (img_normal[:,:,0]-img_min)/(img_max-img_min) * 9 + 1
		img_anomaly = (img_anomaly[:,:,0]-img_min)/(img_max-img_min) * 9 + 1

		img_result = img_anomaly[:,:]-img_normal[:,:]

		img_result = np.reshape(img_result, [int(self.cell_shape[0]/self.move),self.move,int(self.cell_shape[1]/self.move),self.move])
		img_result = img_result[:,0,:,0]
		img_result = np.reshape(img_result, [int(self.cell_shape[0]/self.move),int(self.cell_shape[1]/self.move)])

		return img_result

	def print_eval(self,dir_name, base_img, model_id, thres_hold=0):
		total_anomaly = 0
		# dump code
		# error_list = []
		for file_name in os.listdir(dir_name):
			if os.path.splitext(file_name)[1] != ".png" and os.path.splitext(file_name)[1] != ".jpg" :
				continue
			test = dir_name + "/" + file_name
			result = self.evaluate_img( base_img, test, model_id=model_id,im_show=False,thres_hold=thres_hold)
			# if len(np.where(sub_img > 6)[0]) > 32:
			if result:
				total_anomaly += 1
				# error_list.append(file_name)
			#     print(file_name+ " NG")
			# else:
			#     print(file_name)
			print(file_name," number of anomaly is: ",total_anomaly)
		# with open("dump2.txt","wb") as fp:
		#     pickle.dump(error_list,fp)

	def pre_process(self,pin_normal,list_pin_anomaly):
		test_normal_list = []
		test_anomaly_list = []
		if pin_normal is not None:
			pin_normal = cv2.resize(pin_normal,(self.cell_shape[1],self.cell_shape[0]*9))
		for img_1pin_anomaly in list_pin_anomaly:
			i = 0
			img_1pin_anomaly = cv2.resize(img_1pin_anomaly,(self.cell_shape[1],self.cell_shape[0]*9))
			for y in range(int(pin_normal.shape[0]/self.cell_shape[0])):
				for x in range(int(pin_normal.shape[1]/self.cell_shape[1])):
					cut_img = img_1pin_anomaly[y*self.cell_shape[0]:y*self.cell_shape[0]+self.cell_shape[0],x*self.cell_shape[1]:x*self.cell_shape[1]+self.cell_shape[1]]
					cv2.imwrite("test%02d.png"%i,cut_img)
					test_anomaly_list.append(cut_img)
					test_normal_list.append(pin_normal[y*self.cell_shape[0]:y*self.cell_shape[0]+self.cell_shape[0],x*self.cell_shape[1]:x*self.cell_shape[1]+self.cell_shape[1]])
					cv2.imwrite("compare%02d.png"%i,pin_normal[y*self.cell_shape[0]:y*self.cell_shape[0]+self.cell_shape[0],x*self.cell_shape[1]:x*self.cell_shape[1]+self.cell_shape[1]])
					i+=1
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
		height = self.input_shape[0]
		width = self.input_shape[1]
		number_of_input = (int((self.cell_shape[0]-height)/self.move) + 1)* (int((self.cell_shape[1]-width)/self.move) + 1)  #number of input_shape of each part image
		feed_dict_normal = {}
		feed_dict_anomaly = {}
		#list cut_image of 1 pin (compare image)
		x_sub_normal_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		#list cut_image of 1 pin (test image)
		x_sub_anomaly_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		# preprocess data

		for index, (x_normal, x_anomaly) in enumerate(zip(x_normal_list,x_anomaly_list)):
			x_normal = x_normal.reshape(1,self.cell_shape[0],self.cell_shape[1],self.cell_shape[2])
			x_normal = x_normal / 255
			x_anomaly = x_anomaly.reshape(1,self.cell_shape[0],self.cell_shape[1],self.cell_shape[2])
			x_anomaly = x_anomaly / 255
			for i in range(int((self.cell_shape[0]-height)/self.move)+1):
				for j in range(int((self.cell_shape[1]-width)/self.move)+1):
					x_sub_normal = x_normal[0, i*self.move:i*self.move+height, j*self.move:j*self.move+width, :]
					x_sub_anomaly = x_anomaly[0, i*self.move:i*self.move+height, j*self.move:j*self.move+width, :]
					x_sub_normal = x_sub_normal.reshape(height, width, 3)
					x_sub_anomaly = x_sub_anomaly.reshape( height, width, 3)
					x_sub_normal_list[index % self.number_of_model].append(x_sub_normal)
					x_sub_anomaly_list[index % self.number_of_model].append(x_sub_anomaly)

		for m in range(self.number_of_model):
			feed_dict_normal[self.input_list[m]]  = np.float32(x_sub_normal_list[m])
			feed_dict_anomaly[self.input_list[m]] = np.float32(x_sub_anomaly_list[m])

		# start = time.time()
		#正常のスコア
		loss_list = self.session.run([self.score_list],feed_dict=feed_dict_normal)

		#異常のスコア
		loss_ano_list = self.session.run([self.score_list],feed_dict=feed_dict_anomaly)
		# print("time for inference",time.time() - start)
		result = []

		# start = time.time()
		loss_list = np.array(loss_list[0])
		loss_ano_list = np.array(loss_ano_list[0])

		for index, (loss, loss_ano) in enumerate(zip(loss_list, loss_ano_list)): #number_of_model loop
			for i in range(int(len(loss)/number_of_input)): #number_of_real image loop
				loss_one_img = loss[i*number_of_input:(i+1)*number_of_input]
				loss_ano_one_img = loss_ano[i*number_of_input:(i+1)*number_of_input]
				img_normal, img_anomaly = self.create_loss_img(loss_one_img,loss_ano_one_img)
				#temp
				# result.append(np.amax(self.get_subtraction_score_result(img_normal,img_anomaly)))
				result.append(self.get_subtraction_score_result(img_normal,img_anomaly))
		# print("time for hitmap",time.time() - start)
		return [result[i:num_of_pin*self.number_of_model:num_of_pin] for i in range(num_of_pin)]

	def save_reconstruct_img(self,directory, model_id):
		for file_name in os.listdir(directory):
			print(file_name)
			img = cv2.imread(directory + "/" + file_name)
			img = img / 255
			food = []
			for i in range(int((self.cell_shape[0]-self.input_shape[0])/self.move) +1):
				for j in range(int((self.cell_shape[1]-self.input_shape[1])/self.move) +1):
					food.append(img[i*self.move:i*self.move+self.input_shape[1], j*self.move:j*self.move+self.input_shape[0], :])
			result = self.session.run([self.outputs_mu_list[model_id]],feed_dict={self.input_list[model_id]:np.float32(food)})
			reconstruct = np.zeros([self.cell_shape[0],self.cell_shape[1],3])
			z = 0
			for i in range(int((self.cell_shape[0]-self.input_shape[0])/self.move) +1):
				for j in range(int((self.cell_shape[1]-self.input_shape[1])/self.move) +1):
					reconstruct[i*self.move:i*self.move+self.input_shape[1], j*self.move:j*self.move+self.input_shape[0], :] = result[0][z]
					z+=1

			if not os.path.exists("fukugen"):
				os.makedirs("fukugen")
			cv2.imwrite("fukugen/" + file_name ,reconstruct*255)

if __name__ == "__main__":
	root_direct =  "../20190819_dataset_pin_3_left/"
	# for temp in range(9):
	# 	# temp = 0
	# 	model = Model()
	# 	model.train(
	# 		train_folder= root_direct + 'train_bonus_vae2/%02d' % temp + '/',
	# 		model_path= root_direct + '/model_vae/model',
	# 		data_num=10000,
	# 		number_of_model=9,
	# 		model_id=temp,
	# 		new_combine_model=False,
	# 		resume_single_model=True,
	# 		batch_size=32,
	# 		epochs=5
	# 		)
	# 	del model

	temp = 4
	model = Model()
	model.train(
		train_folder= root_direct + 'train_vae/%02d' % temp + '/',
		model_path= root_direct + '/model_vae/model',
		data_num=100000,
		number_of_model=9,
		model_id=temp,
		new_combine_model=False,
		resume_single_model=False,
		batch_size=32,
		epochs=10
	)
	del model
	# model = Model()
	# model.load_model(model_path= root_direct + 'model_vae/model2',number_of_model=9)

	# test_normal = root_direct + "train_vae/%02d" % temp + "/000000.png"
	# test_anomaly = root_direct + "test_vae_K/%02d" % temp + "/test.png"
	# test_normal = "./compare%02d.png" % temp
	# test_anomaly = "./test%02d.png" % temp
	# #
	# threshold = 6
	# model.print_eval(root_direct + "test_vae/OK/%02d" % temp,test_normal,model_id=temp,thres_hold=threshold)
	# print("***********************************************")
	# model.print_eval(root_direct + "test_vae_K/%02d" % temp,test_normal,model_id=temp,thres_hold=threshold)

	# model.evaluate_img(test_normal, test_anomaly, im_show=True, model_id=temp)

	# model.save_reconstruct_img(root_direct + "test_vae/%02d" % temp  ,temp)
