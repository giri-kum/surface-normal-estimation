import numpy as np
from keras.preprocessing import image

def load_data(data_ID):
	batch_size = len(data_ID)
	train_path = "../eecs442challenge/train/"
	test_path  = "../eecs442challenge/test/"
	ip_path = "color/"
	gt_path = "normal/"
	op_path = "output/"
	mk_path = "mask/"
	X_train = np.empty([1,128,128,1])
	Y_ground = np.empty([1,128,128,4])

	for i in range(batch_size):
	  img1 = image.load_img(train_path + ip_path+str(data_ID[i])+'.png', target_size=(128, 128,3))
	  img2 = image.load_img(train_path + gt_path+str(data_ID[i])+'.png', target_size=(128, 128,3))
	  img3 = image.load_img(train_path + mk_path+str(data_ID[i])+'.png', target_size=(128, 128,3))

	  temp1 = np.reshape((image.img_to_array(img1)).astype('float32')/255,(1,128,128,3))
	  temp2 = np.reshape((image.img_to_array(img2)).astype('float32')/255,(1,128,128,3))
	  temp3 = np.reshape(image.img_to_array(img3)[:,:,0],(1,128,128,1))

	  X_train = np.concatenate((X_train,temp1[:,:,:,:1]),axis=0)
	  temp2 = np.concatenate((temp2,temp3),axis=3)
	  Y_ground = np.concatenate((Y_ground,temp2),axis=0)

	# print("Max raw truth: " + str(np.amax(img2)))

	# print("Input")
	# print("Max: " + str(np.amax(temp1)))
	# print(temp1[0,55:60,55:60,:])
	# print("Truth")
	# print("Max: " + str(np.amax(temp2)))
	# print(temp2[0,55:60,55:60,:])
	# print("Mask")
	# print("Max: " + str(np.amax(temp3)))
	# print(temp3[0,55:60,55:60,:])  

	X_train = X_train[1:,:,:,:]
	Y_ground = Y_ground[1:,:,:,:]
	return [X_train,Y_ground]