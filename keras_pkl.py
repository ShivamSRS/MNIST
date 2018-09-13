import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk 
import keras 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import pickle 
import gzip
import random
# you need to download the mnist.pkl.gzp file from https://www.kaggle.com/brianon99/mnist-data
import os
#using os library change the directory path to the directory where mnist data is stored
#print(os.listdir("../input"))
#os.chdir("../input/")

#unpacking the data
with gzip.open('mnist.pkl.gz') as f:
    train_set, val_set, test_set = pickle.load(f,encoding='latin1') 
train_x, train_y = train_set 
test_x, test_y = test_set 

#normalising data

train_x = train_x.astype('float32')/255
test_x = test_x.astype('float32')/255

print("train images :",train_x.shape)
print("train labels :",train_y.shape)
print("test images :",test_x.shape)
print("test labels :",test_y.shape)
train_x=train_x.reshape(50000,784)
test_x=test_x.reshape(10000,784)

#vectorising the labels
train_y=keras.utils.to_categorical(train_y,10)
test_y=keras.utils.to_categorical(test_y,10)
print(test_y[0])


#defining architecture of neural network
model = Sequential()
model.add(Dense(units=128,activation='relu',input_shape=(784,)))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10, activation = 'softmax'))

#Print a Summary of the model 
model.summary()

#variable learning rate
def lr_schedule(epoch):
    lr = 0.1 
    if epoch > 5: 
        lr = lr/3 
    print("Learning Rate: ",lr) 
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

#optimising and training options
model.compile(optimizer=SGD(lr_schedule(0)),loss="categorical_crossentropy",metrics=["accuracy"])       #learning rate = 0.01

#fitting it
model.fit(train_x,train_y,batch_size=50,epochs=170,shuffle=True,verbose=1,callbacks=[lr_scheduler])

#Evaluate the accuracy of the test dataset
accuracy = model.evaluate(x=test_x,y=test_y,batch_size=50)
print("Accuracy: ",accuracy[1])

#checking a random image to see the result
z=random.randint(1,100)
def visualise_digits(arr_x,z):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    print(z,"th image")
    plt.imshow(arr_x[z].reshape((28,28)),cmap=cm.Greys_r)
    plt.show()
visualise_digits(test_x,z)
img_class = model.predict_classes(test_x) 
classname = img_class[z] 
print("Class: ",classname)
print(test_y[z])
