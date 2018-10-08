
#download dataset frpm neuralnetworksanddeeplearning.com

#MNIST CHARACTER RECOGNITION USING CONVOLUTIONAL NEURAL NETWORKS

#importing the necessary dependencies
import keras 
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.models import Model,Input
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import pickle 
import gzip
import random
# ALTERNATIVELY you can download the mnist.pkl.gzp file from https://www.kaggle.com/brianon99/mnist-data
import os
#using os library change the directory path to the directory where mnist data is stored
#print(os.listdir("../input"))
#os.chdir("../input/")

#unpacking the data
with gzip.open('mnist.pkl.gz') as f:
    train_set, val_set, test_set = pickle.load(f,encoding='latin1') 
train_x, train_y = train_set #50k samples
test_x, test_y = test_set #10k samples
val_x, val_y = val_set #to check the accuracy while fitting

#normalising data
#as computation with large data is tough, making all the pxl values between 0 and 1

train_x = train_x.astype('float32')/255
test_x = test_x.astype('float32')/255

print("train images :",train_x.shape)
print("train labels :",train_y.shape)
print("test images :",test_x.shape)
print("test labels :",test_y.shape)
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
val_x = val_x.reshape(val_x.shape[0], 28, 28, 1)

print("train images :",train_x.shape)

#vectorising the labels
train_y=keras.utils.to_categorical(train_y,10)
test_y=keras.utils.to_categorical(test_y,10)
val_y=keras.utils.to_categorical(val_y,10)
print(val_y[0]) #output is a vector having binary 0 in 9 places and 1 in one place

#defining architecture of neural network
def MiniModel(input_shape):
    images = Input(input_shape) 
    
    net =Conv2D(filters=64,kernel_size=[3,3],strides=[1,1],padding="same",activation="relu")(images) #calling one layer from another ie connecting one layer to another
    net =Conv2D(filters=64,kernel_size=[3,3],strides=[1,1],padding="same",activation="relu")(net)
    net = MaxPooling2D(pool_size=(2,2))(net)
    net =Conv2D(filters=128,kernel_size=[3,3],strides=[1,1],padding="same",activation="relu")(net)
    net = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same",activation="relu")(net)
    
    net = Flatten()(net) #flattening because the last layer needs vectorised data
    net = Dense(units=10,activation="softmax")(net)
    
    model = Model(inputs=images,outputs=net) #using funcrtional api
    return model

input_shape = (28,28,1)
model = MiniModel(input_shape)

model.summary()
#varaible learning rate, needs more optimisation to speed up
def lr_schedule(epoch):
  
 lr = 0.1
 if epoch > 15:
    lr = lr / 100
 elif epoch > 10:
    lr = lr / 10
 elif epoch > 5:
  lr = lr / 5
 print("Learning Rate: ",lr)
 return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

#optimising and training options
#Specify the training components

model.compile(optimizer=SGD(lr_schedule(0)),loss="categorical_crossentropy",metrics=["accuracy"])
#Fit the model

model.fit(train_x,train_y,batch_size=32,epochs=20,shuffle=True,validation_data=[val_x,val_y],verbose=1,callbacks=[lr_scheduler])

#Evaluate the accuracy of the test dataset
accuracy = model.evaluate(x=test_x,y=test_y,batch_size=32)
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

