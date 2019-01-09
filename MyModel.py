import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.models import load_model
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation,BatchNormalization,GlobalAveragePooling2D,Input
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import time
from multiprocessing import Process,Queue
from keras import backend as K
from keras.models import Model
import keras
from keras import regularizers
from keras.regularizers import l2
from utils import*

class MyModel:
	def __init__(self):
		pass

	def VGG(self,num_class=10,image_size=32,channels=3,style='VGG16',weight_decay = 0.0005):
		cfg = {
		    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
		    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
		}
		input_shape=[image_size,image_size,channels]
		model=Sequential()
		
		model.add(Conv2D(64,(3,3),padding='same',
		                 activation='relu',
		                 strides=1,
		                 input_shape=(image_size,image_size,channels),
		                 kernel_regularizer=regularizers.l2(weight_decay)
		                ))
		model.add(BatchNormalization())
		model.add(Dropout(0.3))
		
		guide=cfg[style]
		for i,layer_style in enumerate(guide[1:]):
		    if layer_style == 'M':
		        model.add(MaxPooling2D((2,2)))
		    else:
		        model.add(Conv2D(layer_style,(3,3),padding='same',activation='relu',strides=1,kernel_regularizer=regularizers.l2(weight_decay)))
		        model.add(BatchNormalization())
		        if guide[i+1] != 'M':
		            model.add(Dropout(0.3))
		            
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay),activation='relu'))
		model.add(BatchNormalization())
		
		model.add(Dropout(0.5))
		model.add(Dense(num_class,activation='softmax'))
	#     model.summary()
		return model
    
	def ImprovedCNN(self,num_class=10,image_size=32,channels=3):
		input_shape=[image_size,image_size,channels]
		inputs = Input(shape=input_shape)
		conv=Conv2D(96, (3, 3), activation='relu', padding = 'same', input_shape=(image_size,image_size,channels))
		x=conv(inputs)
		
		x=Dropout(0.2)(x)
		
		x=Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
		x=Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
		x=Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
		x=Dropout(0.5)(x)
		
		x=Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
		x=Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
		x=Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
		x=Dropout(0.5)(x)
		
		x=Conv2D(192, (3, 3), padding = 'same')(x)
		x=Activation('relu')(x)
		x=Conv2D(192, (1, 1),padding='valid')(x)
		x=Activation('relu')(x)
		x=Conv2D(num_class, (1, 1), padding='valid')(x)
		
		x=GlobalAveragePooling2D()(x)
		
		outputs=Activation('softmax')(x)
		model=Model(inputs=inputs, outputs=outputs)		
		return model

	def SimpleCNN(self,num_class=10,image_size=32,channels=3):
		input_shape=[image_size,image_size,channels]
		model=Sequential()
		model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(image_size,image_size,channels	)))
		model.add(Conv2D(32,(3,3),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
		model.add(Conv2D(64,(3,3),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		
		model.add(Flatten())
		
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		
		model.add(Dense(num_class,activation='softmax'))
		
		model.summary()
    
		return model

	def resnet_layer(self,inputs,num_filters=16,kernel_size=3,strides=1,activation='relu',upscale=False):

		conv = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))
		if upscale:
			conv=Conv2D(num_filters,kernel_size=kernel_size,strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))
		x = inputs
		x=conv(x)
		x=BatchNormalization()(x)
		x=Activation(activation)(x)
		conv = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))
		x=conv(x)
		x=BatchNormalization()(x)

		return x
	def shortcut_layer(self,inputs,num_filters=16,kernel_size=1,strides=1,upscale=False):
		if upscale:
			strides=2
		conv = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same')
		y=inputs
		y=conv(y)
		return y

	def ResNet20(self,image_size=32,channels=3,num_class=10,num_filters=16):
		input_shape=[image_size,image_size,channels]
		inputs = Input(shape=input_shape)
		resBlock_num=3
		stack=3
		filter_num=[16,32,64]

		x=Conv2D(16,(3,3),strides=1,padding='same')(inputs)
		x=BatchNormalization()(x)
		x=Activation('relu')(x)
		
		for i in range(stack):
			for j in range(resBlock_num):
				# print(i,j)
				if j==0 and i!=0:
					y=self.resnet_layer(x,num_filters=filter_num[i],upscale=True)
					x=self.shortcut_layer(x,upscale=True,num_filters=filter_num[i])
				else:
					y=self.resnet_layer(x,num_filters=filter_num[i])
					x=self.shortcut_layer(x,num_filters=filter_num[i])
				x=keras.layers.add([x,y])
				x=Activation('relu')(x)

		x=BatchNormalization()(x)
		x=Activation('relu')(x)
		x=GlobalAveragePooling2D()(x)
		# x = Flatten()(x)
		outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)
		model = Model(inputs=inputs, outputs=outputs)
		# model.summary()
		return model

	def ResNet18(self,image_size=32,channels=3,num_class=10,num_filters=16):
		input_shape=[image_size,image_size,channels]
		inputs = Input(shape=input_shape)
		x=Conv2D(16,(3,3),strides=1,padding='same')(inputs)

