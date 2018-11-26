import numpy as np
from keras.preprocessing import image
import keras.backend as K
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt
from my_load import load_data
from my_loss import compute_MAE
#################################################################################################################################
op_path = "../eecs442challenge/train/debug/"
data_path = "../data/"

perform_validation = True
total_train_size = 20000 #500
no_of_validation_images = 2000
train_size = total_train_size - no_of_validation_images
validation_rate = 100
validation_losses = []

batch_size = 16  #32
start_epoch = 0
no_of_epochs = 1
#2
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

def save_loss(losses, is_validation):
  data = '\n'.join(str(z) for z in losses)
  if(is_validation):
    file = open("../data/validation_loss.csv", "w")
  else:
    file = open("../data/loss.csv", "w")
  file.write(data)
  file.close()

################################

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
saver = tf.train.Saver()
data_ID = np.arange(train_size)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
iterations = np.int64(np.floor(train_size/batch_size))
losses = []
min_loss = .02
if(start_epoch != 0):
  model_path = "../data/model" + str(start_epoch-1) + ".ckpt"
#  tf.reset_default_graph()
  saver.restore(sess, model_path) #Change this model id model<x>.ckpt x is 1,2,3.. 
  print("Model restored")    

for epoch in range(no_of_epochs):
  np.random.shuffle(data_ID)
  for i in range(iterations):
    batch_data_ID = data_ID[i*batch_size:(i+1)*batch_size]

    [X_train,Y_ground] = load_data(batch_data_ID)
    mode = True
    [eval_train_step, eval_loss, output] = sess.run([train_step, loss, prediction],feed_dict={x_train: X_train, y_ground: Y_ground, training_mode: mode})

    losses.append(eval_loss)
    try:
      save_loss(losses, False)
    except:
      pass

    print('epoch: '+ str(epoch) + ' of ' + str(no_of_epochs-1),'iteration: '+str(i) + ' of ' + str(iterations-1),'loss = ' + str(eval_loss)) #+' and '+str(check_loss))
    
    if( (i % (validation_rate-1) == 0) and perform_validation):
      temp = 0.0
      for j in range(no_of_validation_images):
        [X_train,Y_ground] = load_data([train_size+j])
        temp  = temp + sess.run(loss,feed_dict={x_train: X_train, y_ground: Y_ground, training_mode: False})
      validation_losses.append(temp/no_of_validation_images)
      print('validation loss: ', validation_losses[-1], ' last loss out of ', len(validation_losses))
      save_loss(validation_losses, True)

    scipy.misc.toimage(output[0,:,:,:]).save(op_path+str(epoch+start_epoch)+'_'+str(i)+'.png') #Automatically rescales images to use complete 0 to 255 range.
    if(i == iterations-1):
      save_path = saver.save(sess, "../data/"+"model"+str(epoch+start_epoch)+".ckpt")
      print("Model saved in path: %s" % save_path)
    if(eval_loss < min_loss):
      wr = open("../data/model_bestTrainingLoss.ckpt", 'w')
      min_loss = eval_loss
      print('New min training loss: '+ str(min_loss))
      save_path = saver.save(sess, "../data/model_bestTrainingLoss.ckpt")
sess.close()

print('Min training loss: ' + str(min_loss))



