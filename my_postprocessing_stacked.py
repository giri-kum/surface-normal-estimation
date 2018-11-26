import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from matplotlib import pyplot as plt
import keras.backend as K
import scipy.misc
import sys

perform_validation = True
validation_rate = 100

def get_my_model(batch_size):
	#################################################################################################################################
	
	learning_rate = .0001
	alpha_lr = 0.3

	#################################################################################################################################

	x_train = tf.placeholder(tf.float32, shape=[None, 128,128,1])
	y_ground = tf.placeholder(tf.float32, shape=[None, 128,128,4])
	training_mode = tf.placeholder(tf.bool)

	def conv2d(x, W):
	  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	def max_pool2d(x):
	  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
	                        strides=[1, 2, 2, 1], padding='SAME')
	def weight_variable(shape):
	  initial = tf.truncated_normal(shape, stddev=0.1)
	  return tf.Variable(initial, trainable = True)

	def bias_variable(shape):
	  initial = tf.constant(0.1, shape=shape)
	  return tf.Variable(initial, trainable = True)



	# set 1 - 128x128x64 ===================================================
	# input: 128x128x1

	# conv 1x1 ----------------
	S1_W1_BN = weight_variable([1,1,1,128])
	S1_conv1_BN = conv2d(x_train, S1_W1_BN)
	# BN
	S1_bn1 = tf.layers.batch_normalization(S1_conv1_BN, training = training_mode)

	# Relu ---------------------
	S1_relu1 = tf.nn.leaky_relu(S1_bn1, alpha=alpha_lr)

	# conv 3x3 ----------------
	S1_W2_BN = weight_variable([3,3,128,128])
	S1_conv2_BN = conv2d(S1_relu1, S1_W2_BN)
	# BN
	S1_bn2 = tf.layers.batch_normalization(S1_conv2_BN, training = training_mode)

	# Relu ---------------------
	S1_relu2 = tf.nn.leaky_relu(S1_bn2, alpha=alpha_lr)
	# left with: 128x128x64

	# set 2 - 64x64x128 ====================================================
	# input: 128x128x64
	# pool
	S2_pool1 = max_pool2d(S1_relu2)

	# conv 1x1 -----------------
	S2_W1_BN = weight_variable([1,1,128,256])
	S2_conv1_BN = conv2d(S2_pool1, S2_W1_BN)
	# BN
	S2_bn1 = tf.layers.batch_normalization(S2_conv1_BN, training = training_mode)

	# Relu ---------------------
	S2_relu1 = tf.nn.leaky_relu(S2_bn1, alpha=alpha_lr)

	# conv 3x3 ------------------
	S2_W2_BN = weight_variable([3,3,256,256])
	S2_conv2_BN = conv2d(S2_relu1, S2_W2_BN)
	# BN
	S2_bn2 = tf.layers.batch_normalization(S2_conv2_BN, training = training_mode)

	# Relu ---------------------
	S2_relu2 = tf.nn.leaky_relu(S2_bn2, alpha=alpha_lr)
	# left with 64x64x128

	# set 3 - 32x32x256 ====================================================
	# input: 64x64x128

	# pool
	S3_pool1 = max_pool2d(S2_relu2)

	# conv 1x1 -----------------
	S3_W1_BN = weight_variable([1,1,256,512])
	S3_conv1_BN = conv2d(S3_pool1, S3_W1_BN)
	# BN
	S3_bn1 = tf.layers.batch_normalization(S3_conv1_BN, training = training_mode)

	# Relu ---------------------
	S3_relu1 = tf.nn.leaky_relu(S3_bn1, alpha=alpha_lr)

	# conv 3x3 ------------------
	S3_W2_BN = weight_variable([3,3,512,512])
	S3_conv2_BN = conv2d(S3_relu1, S3_W2_BN)
	# BN
	S3_bn2 = tf.layers.batch_normalization(S3_conv2_BN, training = training_mode)

	# Relu ---------------------
	S3_relu2 = tf.nn.leaky_relu(S3_bn2, alpha=alpha_lr)
	# left with 64x64x128

	# set 2 - 64x64x128 ===================================================
	# input: 32x32x256
	# conv 1x1 -----------------
	S2_W3_BN = weight_variable([1,1,512,256])
	S2_conv3_BN = conv2d(S3_relu2, S2_W3_BN)
	# BN
	S2_bn3 = tf.layers.batch_normalization(S2_conv3_BN, training = training_mode)
	# # upsample last result
	# S2_conv3_BN_shape = S2_conv3_BN.get_shape().as_list()
	# S2_upsample3 = tf.image.resize_images(S2_conv3_BN, [2*S2_conv3_BN_shape[1],2*S2_conv3_BN_shape[2]],tf.image.ResizeMethod.BILINEAR) 
	# upsample last result
	S2_bn3_shape = S2_bn3.get_shape().as_list()
	S2_upsample3 = tf.image.resize_images(S2_bn3, [2*S2_bn3_shape[1],2*S2_bn3_shape[2]],tf.image.ResizeMethod.BILINEAR) 
	# # BN
	# S2_batch_mean3, S2_batch_var3 = tf.nn.moments(S2_upsample3, [0])
	# S2_bn3 = (S2_upsample3 - S2_batch_mean3)/tf.sqrt(S2_batch_var3 + epsilon)
	# concat
	S2_concat3 = tf.concat([S2_relu2,S2_upsample3],-1)

	# relu ---------------------
	S2_relu3 = tf.nn.leaky_relu(S2_concat3, alpha=alpha_lr)

	# conv 1x1 ------------------
	S2_W4_BN = weight_variable([1,1,512,256])
	S2_conv4_BN = conv2d(S2_relu3, S2_W4_BN)
	# BN
	S2_bn4 = tf.layers.batch_normalization(S2_conv4_BN, training = training_mode)

	# Relu ---------------------
	S2_relu4 = tf.nn.leaky_relu(S2_bn4, alpha=alpha_lr)

	# conv 3x3 -------------------
	S2_W5_BN = weight_variable([3,3,256,256])
	S2_conv5_BN = conv2d(S2_relu4, S2_W5_BN)
	# BN
	S2_bn5 = tf.layers.batch_normalization(S2_conv5_BN, training = training_mode)

	# Relu ---------------------
	S2_relu5 = tf.nn.leaky_relu(S2_bn5, alpha=alpha_lr)
	# left with 64x64x128

	# set 1 - 128x128x64 ==================================================
	# input: 64x64x128
	# conv 1x1 ------------------
	S1_W3_BN = weight_variable([1,1,256,128])
	S1_conv3_BN = conv2d(S2_relu5, S1_W3_BN)
	# BN
	S1_bn3 = tf.layers.batch_normalization(S1_conv3_BN, training = training_mode)
	# upsample last result
	S1_bn3_shape = S1_bn3.get_shape().as_list()
	S1_upsample3 = tf.image.resize_images(S1_bn3, [2*S1_bn3_shape[1],2*S1_bn3_shape[2]],tf.image.ResizeMethod.BILINEAR) 
	# concat
	S1_concat3 = tf.concat([S1_relu2,S1_upsample3], -1)

	# relu ---------------------
	S1_relu3 = tf.nn.leaky_relu(S1_concat3, alpha=alpha_lr)

	# conv 1x1 -----------------
	S1_W4_BN = weight_variable([1,1,256,128])
	S1_conv4_BN = conv2d(S1_relu3, S1_W4_BN)
	# BN
	S1_bn4 = tf.layers.batch_normalization(S1_conv4_BN, training = training_mode)

	# Relu ---------------------
	S1_relu4 = tf.nn.leaky_relu(S1_bn4, alpha=alpha_lr)

	# conv 3x3 ------------------
	S1_W5_BN = weight_variable([3,3,128,128])
	S1_conv5_BN = conv2d(S1_relu4, S1_W5_BN)
	# BN
	S1_bn5 = tf.layers.batch_normalization(S1_conv5_BN, training = training_mode)

	# Relu ---------------------
	S1_relu5 = tf.nn.leaky_relu(S1_bn5, alpha=alpha_lr)
	# left with 128x128x64

	# dropout = tf.layers.dropout(S1_relu5, rate = 0.4, training = training_mode)

	# final output - 128x128x3
	# input: 128x128x64
	# conv 1x1
	pred_W1 = weight_variable([1,1,128,3])

	prediction_1 = conv2d(S1_relu5, pred_W1)
	# left with 128x128x3

	##########STACK2##########################

	# set 1 - 128x128x64 ===================================================
	# input: 128x128x3

	# conv 1x1 ----------------
	SS1_W1_BN = weight_variable([1,1,3,128])
	SS1_conv1_BN = conv2d(prediction_1, SS1_W1_BN)
	# BN
	SS1_bn1 = tf.layers.batch_normalization(SS1_conv1_BN, training = training_mode)

	# Relu ---------------------
	SS1_relu1 = tf.nn.leaky_relu(SS1_bn1, alpha=alpha_lr)

	# conv 3x3 ----------------
	SS1_W2_BN = weight_variable([3,3,128,128])
	SS1_conv2_BN = conv2d(SS1_relu1, SS1_W2_BN)
	# BN
	SS1_bn2 = tf.layers.batch_normalization(SS1_conv2_BN, training = training_mode)

	# Relu ---------------------
	SS1_relu2 = tf.nn.leaky_relu(SS1_bn2, alpha=alpha_lr)
	# left with: 128x128x64

	# set 2 - 64x64x128 ====================================================
	# input: 128x128x64
	# pool
	SS2_pool1 = max_pool2d(SS1_relu2)

	# conv 1x1 -----------------
	SS2_W1_BN = weight_variable([1,1,128,256])
	SS2_conv1_BN = conv2d(SS2_pool1, SS2_W1_BN)
	# BN
	SS2_bn1 = tf.layers.batch_normalization(SS2_conv1_BN, training = training_mode)

	# Relu ---------------------
	SS2_relu1 = tf.nn.leaky_relu(SS2_bn1, alpha=alpha_lr)

	# conv 3x3 ------------------
	SS2_W2_BN = weight_variable([3,3,256,256])
	SS2_conv2_BN = conv2d(SS2_relu1, SS2_W2_BN)
	# BN
	SS2_bn2 = tf.layers.batch_normalization(SS2_conv2_BN, training = training_mode)

	# Relu ---------------------
	SS2_relu2 = tf.nn.leaky_relu(SS2_bn2, alpha=alpha_lr)
	# left with 64x64x128

	# set 3 - 32x32x256 ====================================================
	# input: 64x64x128

	# pool
	SS3_pool1 = max_pool2d(SS2_relu2)

	# conv 1x1 -----------------
	SS3_W1_BN = weight_variable([1,1,256,512])
	SS3_conv1_BN = conv2d(SS3_pool1, SS3_W1_BN)
	# BN
	SS3_bn1 = tf.layers.batch_normalization(SS3_conv1_BN, training = training_mode)

	# Relu ---------------------
	SS3_relu1 = tf.nn.leaky_relu(SS3_bn1, alpha=alpha_lr)

	# conv 3x3 ------------------
	SS3_W2_BN = weight_variable([3,3,512,512])
	SS3_conv2_BN = conv2d(SS3_relu1, SS3_W2_BN)
	# BN
	SS3_bn2 = tf.layers.batch_normalization(SS3_conv2_BN, training = training_mode)

	# Relu ---------------------
	SS3_relu2 = tf.nn.leaky_relu(SS3_bn2, alpha=alpha_lr)
	# left with 64x64x128

	# set 2 - 64x64x128 ===================================================
	# input: 32x32x256
	# conv 1x1 -----------------
	SS2_W3_BN = weight_variable([1,1,512,256])
	SS2_conv3_BN = conv2d(SS3_relu2, SS2_W3_BN)
	# BN
	SS2_bn3 = tf.layers.batch_normalization(SS2_conv3_BN, training = training_mode)
	# # upsample last result
	# SS2_conv3_BN_shape = SS2_conv3_BN.get_shape().as_list()
	# SS2_upsample3 = tf.image.resize_images(SS2_conv3_BN, [2*SS2_conv3_BN_shape[1],2*SS2_conv3_BN_shape[2]],tf.image.ResizeMethod.BILINEAR) 
	# upsample last result
	SS2_bn3_shape = SS2_bn3.get_shape().as_list()
	SS2_upsample3 = tf.image.resize_images(SS2_bn3, [2*SS2_bn3_shape[1],2*SS2_bn3_shape[2]],tf.image.ResizeMethod.BILINEAR) 
	# # BN
	# SS2_batch_mean3, SS2_batch_var3 = tf.nn.moments(SS2_upsample3, [0])
	# SS2_bn3 = (SS2_upsample3 - SS2_batch_mean3)/tf.sqrt(SS2_batch_var3 + epsilon)
	# concat
	SS2_concat3 = tf.concat([SS2_relu2,SS2_upsample3],-1)

	# relu ---------------------
	SS2_relu3 = tf.nn.leaky_relu(SS2_concat3, alpha=alpha_lr)

	# conv 1x1 ------------------
	SS2_W4_BN = weight_variable([1,1,512,256])
	SS2_conv4_BN = conv2d(SS2_relu3, SS2_W4_BN)
	# BN
	SS2_bn4 = tf.layers.batch_normalization(SS2_conv4_BN, training = training_mode)

	# Relu ---------------------
	SS2_relu4 = tf.nn.leaky_relu(SS2_bn4, alpha=alpha_lr)

	# conv 3x3 -------------------
	SS2_W5_BN = weight_variable([3,3,256,256])
	SS2_conv5_BN = conv2d(SS2_relu4, SS2_W5_BN)
	# BN
	SS2_bn5 = tf.layers.batch_normalization(SS2_conv5_BN, training = training_mode)

	# Relu ---------------------
	SS2_relu5 = tf.nn.leaky_relu(SS2_bn5, alpha=alpha_lr)
	# left with 64x64x128

	# set 1 - 128x128x64 ==================================================
	# input: 64x64x128
	# conv 1x1 ------------------
	SS1_W3_BN = weight_variable([1,1,256,128])
	SS1_conv3_BN = conv2d(SS2_relu5, SS1_W3_BN)
	# BN
	SS1_bn3 = tf.layers.batch_normalization(SS1_conv3_BN, training = training_mode)
	# upsample last result
	SS1_bn3_shape = SS1_bn3.get_shape().as_list()
	SS1_upsample3 = tf.image.resize_images(SS1_bn3, [2*SS1_bn3_shape[1],2*SS1_bn3_shape[2]],tf.image.ResizeMethod.BILINEAR) 
	# concat
	SS1_concat3 = tf.concat([SS1_relu2,SS1_upsample3], -1)

	# relu ---------------------
	SS1_relu3 = tf.nn.leaky_relu(SS1_concat3, alpha=alpha_lr)

	# conv 1x1 -----------------
	SS1_W4_BN = weight_variable([1,1,256,128])
	SS1_conv4_BN = conv2d(SS1_relu3, SS1_W4_BN)
	# BN
	SS1_bn4 = tf.layers.batch_normalization(SS1_conv4_BN, training = training_mode)

	# Relu ---------------------
	SS1_relu4 = tf.nn.leaky_relu(SS1_bn4, alpha=alpha_lr)

	# conv 3x3 ------------------
	SS1_W5_BN = weight_variable([3,3,128,128])
	SS1_conv5_BN = conv2d(SS1_relu4, SS1_W5_BN)
	# BN
	SS1_bn5 = tf.layers.batch_normalization(SS1_conv5_BN, training = training_mode)

	# Relu ---------------------
	SS1_relu5 = tf.nn.leaky_relu(SS1_bn5, alpha=alpha_lr)
	# left with 128x128x64

	# dropout = tf.layers.dropout(SS1_relu5, rate = 0.4, training = training_mode)

	# final output - 128x128x3
	# input: 128x128x64
	# conv 1x1
	pred_W = weight_variable([1,1,128,3])

	prediction = conv2d(SS1_relu5, pred_W)
	# left with 128x128x3

	#################################################################################################################################

	x_mask = y_ground[:,:,:,3]
	new_y_ground = y_ground[:,:,:,:3]
	new_x_mask = x_mask != 0
	new_x_mask = K.cast(new_x_mask, dtype = tf.bool)
	masked_prediction = K.switch(new_x_mask,prediction,K.zeros_like(prediction))
	masked_y_ground = K.switch(new_x_mask,new_y_ground,K.zeros_like(new_y_ground))
	loss = tf.reduce_mean(tf.square( masked_y_ground - masked_prediction )) #compute_MAE(masked_prediction,masked_y_ground,x_mask, batch_size)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
	  train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	#################################################################################################################################

	return [x_train, y_ground, training_mode, prediction, loss,train_step]

def run_test(test_data,test_size):
	paths  = ["../eecs442challenge/train/","../eecs442challenge/test/"]
	test_path = paths[test_data]
	tf.reset_default_graph()
	ip_path = "color/"
	op_path = "output/"

	X_test = np.empty([1,128,128,1])
	Y_ground = np.empty([1,128,128,4])
	batch_size = 1

	X_stats = np.empty([1,128,128,1])

	# Load in images to get overall mean and variance
	# filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(test_path + "*.png"))

	# image_reader = tf.WholeFileReader()

	# _, image_file = image_reader.read(filename_queue)

	# imageTensor = tf.image.decode_png(image_file)

	# imageTensor = tf.cast(imageTensor, tf.float32)

	# print('Type: ' + str(imageTensor.dtype))
	# pop_mean, pop_var = tf.nn.moments(imageTensor, [0])


	# for i in range(test_size):
	# 	print("processing image ", i,end='\r')
	# 	img1 = image.load_img(test_path + ip_path+str(i)+'.png', target_size=(128,128,3))
	# 	temp1 = np.reshape((image.img_to_array(img1)).astype('float32')/255,(1,128,128,3))
	# 	X_stats = np.concatenate((X_test,temp1[:,:,:,:1]),axis=0)
	# 	X_stats = X_test[1:,:,:,:]

	
	# pop_mean = np.mean(X_stats, axis = 0)
	# pop_var = np.var(X_stats, axis = 0)

	# print('Mean: ' + str(pop_mean))
	# print('Var: ' + str(pop_var))

	[x_train, y_ground, training_mode, prediction, loss,train_step] = get_my_model(batch_size)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, "../data/model_bestTrainingLoss.ckpt") #Change this model id model<x>.ckpt x is 1,2,3.. 
		print("Model restored")
		for i in range(test_size):
			print("processing image ", i,end='\r')
			img1 = image.load_img(test_path + ip_path+str(i)+'.png', target_size=(128,128,3))
			temp1 = np.reshape((image.img_to_array(img1)).astype('float32')/255,(1,128,128,3))
			X_test = np.concatenate((X_test,temp1[:,:,:,:1]),axis=0)
			X_test = X_test[1:,:,:,:] 
			mode = False
			output  = prediction.eval(feed_dict={x_train: X_test, y_ground: Y_ground, training_mode: mode})
			scipy.misc.toimage(output[0,:,:,:]).save(test_path+op_path+str(i)+'.png')	
try:
	def get_loss(path):
		file = open(path, "r")
		data = file.read();
		lines = data.splitlines();
		loss = [float(i) for i in lines]
		return loss
	new_loss = get_loss("../data/loss.csv")
	plt.plot(range(len(new_loss[1:])), new_loss[1:], '.-', label = 'training') #Plotting leaving the first sample, because it always huge and changes the scale of the plot
	if(perform_validation):
		validation_loss = get_loss("../data/validation_loss.csv")
		plt.plot([validation_rate*k for k in range(len(validation_loss[1:]))], validation_loss[1:], 'x-', label = 'validation')
	plt.legend(loc='upper right')
	plt.ylabel('loss')
	plt.xlabel('iteration')
	plt.show()
except:
	print("loss could not be plotted")

run_test(1,2000) #To run training data run_test(0,num_images)
