from __future__ import unicode_literals
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation,GlobalAveragePooling2D,Input
from keras.optimizers import Adam
from keras.models import Model
from utils import *
from MyModel import MyModel
import twitter
import sys


# Twitter-related API
def getQ(li):
    return '%23'+'%20OR%20%23'.join(li)
def getWeightedTag(results):
    tags={}
    for status in results:
        for tag in status.hashtags:
            text=tag.text
            tags[text]=tags.get(text,0)+1
    return tags
def getGoodTags(tagsDict):
    sortedTags=sorted(tagsDict.items(),key=lambda x:-x[1])
    base=[pair[0] for pair in sortedTags[:3]]
    good_tags=[]
    for pair in sortedTags:
        if pair[1]>1:
            good_tags+=[pair[0]]
        else:
            for word in base:
                if pair[0].lower() in word.lower() or word.lower() in pair[0].lower():
                    good_tags+=[pair[0]]
    return good_tags
def generateQuery(hashtag,result_type='recent',count=100,lang='en'):
    return "q=%23{}&result_type={}&count={}&lang={}".format(hashtag,result_type,count,lang)

def read_and_resize(input_path,mean):
    im = Image.open(input_path)
    small_pic=im.resize((32,32))
    small_pic=np.array(small_pic,dtype=float)/255.0
    return small_pic

train_image,test_image,train_label_onehot,test_label_onehot,class_names,mean=get_data(10)

ModelPool= MyModel()
model=ModelPool.VGG()
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy'])
model.load_weights('vgg16.h5')

input_path=sys.argv[1]
x=np.array([read_and_resize("images/"+input_path,mean)])

pred=model.predict(x=x)[0]
ans=np.argmax(pred)
class_pred=class_names[ans]

api = twitter.Api(consumer_key='YOUR CONSUMER KEY',
                  consumer_secret='YOUR CONSUMER SECRET',
                  access_token_key='YOUR TOKEN KEY',
                  access_token_secret='YOUR TOKEN SECRET')

query=generateQuery(class_pred)
results = api.GetSearch(raw_query=query)
tags=getWeightedTag(results)
good_tags=getGoodTags(tags)
print('I guess this is a: ',class_pred)
output=""
for tag in good_tags:
    output+="#"+tag+"   "
print(output)