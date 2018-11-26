import numpy as np
import keras.backend as K
import tensorflow as tf
def compute_mse(batch_prediction,batch_groundtruth,batch_size):
    batch_mask = batch_groundtruth[:,:,:,3]
    batch_groundtruth = batch_groundtruth[:,:,:,0:3]; #0:3 gives index from 0 to 3 excluding 3!
    batch_prediction = batch_prediction[:,:,:,0:3]
    loss = tf.losses.mean_squared_error(batch_groundtruth,batch_prediction)
        
    return loss


def compute_MAE(batch_prediction,batch_groundtruth,batch_size):
    batch_mask = batch_groundtruth[:,:,:,3]
    batch_groundtruth = batch_groundtruth[:,:,:,0:3]; #0:3 gives index from 0 to 3 excluding 3!
    batch_prediction = batch_prediction[:,:,:,0:3]
#    batch_groundtruth = batch_prediction
#    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_groundtruth, logits=batch_prediction))
    
    pshape = batch_prediction.get_shape().as_list()
    gshape = batch_groundtruth.get_shape().as_list()
    mshape = batch_mask.get_shape().as_list()

#    print('from compute_loss: prediction shape = ',pshape)
#    print('from compute_loss: groundtruth shape = ',gshape)   
#    print('from compute_loss: mask shape = ',mshape) 
    mean_angle_error = tf.Variable(0.0,dtype = 'float32')
#    print(mean_angle_error.get_shape().as_list())       
    total_pixels = tf.Variable(0,dtype = 'int64')
    for i in range(batch_size):
        mask = batch_mask[i,:,:]
        prediction = ((batch_prediction[i,:,:,:] / 255.0) - 0.5) * 2
        groundtruth = ((batch_groundtruth[i,:,:,:] / 255.0) - 0.5) * 2
        pixels = tf.count_nonzero(mask)
        mask = mask != 0
        mask = K.cast(mask, dtype = tf.bool)
        temp11 = K.sum(prediction * prediction, axis=2)
        temp22 = K.sum(groundtruth * groundtruth, axis=2)
        temp12 = K.sum(prediction * groundtruth, axis=2)
        a11 = K.switch(mask,temp11,K.zeros_like(temp11))
        a22 = K.switch(mask,temp22,K.zeros_like(temp22))
        a12 = K.switch(mask,temp12,K.zeros_like(temp12))
        a11 = K.reshape(a11,[-1])
        a22 = K.reshape(a22,[-1])
        a12 = K.reshape(a12,[-1])
        cos_dist = a12 #/ K.sqrt(a11 * a22)
        neg_ones = -1*K.ones_like(cos_dist)
        nan_locations = tf.is_nan(cos_dist)
        cos_dist = K.switch(nan_locations,cos_dist,neg_ones)
        cos_dist = K.clip(cos_dist, -1, 1)
        angle_error = tf.acos(cos_dist)#np.arccos(cos_dist) #hope both acos and arccos are the same
        mean_angle_error = tf.add(K.sum(angle_error),mean_angle_error)
#        print(mean_angle_error)
#        mean_angle_error = tf.acos(tf.cos(mean_angle_error))
        total_pixels = tf.add(pixels,total_pixels)
    loss = mean_angle_error/ K.cast(total_pixels, dtype = tf.float32)
    

    return loss
