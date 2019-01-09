import matplotlib
from matplotlib import pyplot as plt
import  tensorflow as tf
from keras.utils import np_utils
import numpy as np

def plot_model(model_details,name):

    # Create sub-plots
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # Summarize history for accuracy
    axs[0].plot(range(1,len(model_details.history['acc'])+1),model_details.history['acc'])
    axs[0].plot(range(1,len(model_details.history['val_acc'])+1),model_details.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['acc'])+1),len(model_details.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    
    # Summarize history for loss
    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    
    # Show the plot
    fig.savefig(name,dpi=300)
    plt.show()

def get_data(num=10):
	IMAGE_SIZE = 32
	CHANNELS = 3
	NUM_CLASSES=num
	batch_size=128

	if NUM_CLASSES==10:
		(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()
		class_names=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
	else:
		(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar100.load_data(label_mode='fine')
		class_names=['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm']

	train_label_onehot=np_utils.to_categorical(y_train,num_classes=NUM_CLASSES)
	test_label_onehot=np_utils.to_categorical(y_test,num_classes=NUM_CLASSES)

	train_image=np.array(x_train,dtype=float)/255.0
	test_image=np.array(x_test,dtype=float)/255.0
	mean=np.mean(train_image)
	train_image-=mean
	test_image-=mean
	return train_image,test_image,train_label_onehot,test_label_onehot,class_names,mean