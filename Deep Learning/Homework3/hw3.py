# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:57:28 2019
@author: willywanghw
"""
#7108029211 王皓威

#Hw3 Cnn卷積神經網路
import keras
keras.__version__
from keras import layers 
from keras import models
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)
test_labels

import matplotlib.pyplot as plt 
from keras.utils import to_categorical


train_images = train_images.reshape((60000, 28,28,1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28,28,1)) 
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#optimizerlist = ['rmsprop','Adam','SGD','Adagrad','Adadelta','Adamax','Nadam']
#b_size = [64,128,256,512]
#ep = [5,10,25]
#for opt in optimizerlist : 
#for b in b_size:
#for e in ep :
#    plt.clf()
CnnModel = models.Sequential()

CnnModel.add(layers.Conv2D(32,(3,3),activation = 'relu',
                                  input_shape =(28,28,1)))

CnnModel.add(layers.MaxPooling2D((2,2)))
CnnModel.add(layers.Conv2D(128,(3,3),activation = 'relu'))
CnnModel.add(layers.MaxPooling2D(2,2))
CnnModel.add(layers.Conv2D(64,(3,3),activation = 'relu'))
CnnModel.add(layers.Flatten())
CnnModel.add(layers.Dense(64,activation = 'relu'))
CnnModel.add(layers.Dense(10,activation = 'softmax'))
CnnModel.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
#    networkht = network.fit(train_images, train_labels, epochs=e, batch_size=256,validation_data=(test_images,test_labels), verbose=1)
networkht = CnnModel.fit(train_images, train_labels
                        , epochs=5, batch_size=64,validation_data=(test_images,test_labels), verbose=1)
net_dict = networkht.history
test_loss, test_acc = CnnModel.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
loss = networkht.history['loss']
val_loss = networkht.history['val_loss']
epochs = range(1,len(loss)+1)    
plt.plot(epochs,loss,'r',label='Training loss')
plt.plot(epochs,val_loss,'b',label = 'validation loss')
#    plt.title('Training and validataion loss Epochs = '+str(e))
#    plt.title('Training and validataion loss batch_size = '+str(b))
plt.title('Cnn Training and validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#    plt.savefig('loss epoch = '+str(e))
#    plt.savefig('loss batch_size = '+str(b))
plt.savefig('loss')
plt.show()
plt.clf()
acc = networkht.history['accuracy']
val_acc = networkht.history['val_accuracy']
plt.plot(epochs,acc,'r',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
#    plt.title('Training and Validation accuracy Epochs = '+str(e))
#    plt.title('Training and Validation accuracy batch_size = '+str(b))
plt.title('Cnn Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#    plt.savefig('acc epoch = '+str(e))
#    plt.savefig('acc batch_size = '+str(b))
plt.savefig('acc')
plt.show()