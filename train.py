import numpy as np
import tensorflow as tf
import time
import keras

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.models import load_model
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation,BatchNormalization,GlobalAveragePooling2D,Input
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Process,Queue
from keras import backend as K
from keras.models import Model
from keras import regularizers
from utils import*
from MyModel import MyModel

def train(x,y,valid_x,valid_y,save_name,model,lr=1e-4,lr_drop=0,decay=1e-6,epochs=250,lr_reduce_factor=0.2,batch_size=128):
	model.summary()
	checkpoint = ModelCheckpoint(save_name,monitor= 'val_acc', verbose = 1, save_best_only = True,mode='auto')
	lr_reducer=ReduceLROnPlateau(monitor='val_acc', factor=lr_reduce_factor, patience=4, min_lr=1e-6,verbose=1)
	callbacks=[checkpoint,lr_reducer]
	model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr,decay=decay),metrics=['accuracy'])
	datagen = ImageDataGenerator(
		featurewise_center=False,
		samplewise_center=False,
		featurewise_std_normalization=False,# divide each input by its std
		samplewise_std_normalization=False,# apply ZCA whitening
		zca_whitening=False,# epsilon for ZCA whitening
		zca_epsilon=1e-06,# randomly rotate images in the range (deg 0 to 180)
		rotation_range=0,# randomly shift images horizontally
		width_shift_range=0.1,# randomly shift images vertically
		height_shift_range=0.1,# set range for random shear
		shear_range=0.,# set range for random zoom
		zoom_range=0.,# set range for random channel shifts
		channel_shift_range=0.,# set mode for filling points outside the input boundaries
		fill_mode='nearest',# value used for fill_mode = "constant"
		cval=0.,# randomly flip images
		horizontal_flip=True,# randomly flip images
		vertical_flip=False,# set rescaling factor (applied before any other transformation)
		rescale=None,# set function that will be applied on each input
		preprocessing_function=None,# image data format, either "channels_first" or "channels_last"
		data_format=None,# fraction of images reserved for validation (strictly between 0 and 1)
		validation_split=0.0)
	datagen.fit(x)

	time1=time.time()
	model.fit_generator(datagen.flow(x, y, batch_size=batch_size),
                    steps_per_epoch=  x.shape[0]//batch_size,
                        validation_data=(valid_x, valid_y),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)
	time2=time.time()
	print("TOTAL TIME:{} s".format(round(time2-time1),3))
	return model

train_image,test_image,train_label_onehot,test_label_onehot,class_names=get_data(10)
ModelPool = MyModel();
model=ModelPool.VGG(num_class=10)
# ['SimpleCNN','ImprovedCNN','VGG','ResNet20']
train(model=model,x=train_image,y=train_label_onehot,valid_x=test_image,
	valid_y=test_label_onehot,save_name='vgg_cifar10.py')
