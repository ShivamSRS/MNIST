import numpy as np # linear algebra


import keras 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import SGD
from keras.datasets import mnist

#load the mnist dataset
(train_x, train_y) , (test_x, test_y) = mnist.load_data() #may take a few minutes to download
#normalize the data

train_x = train_x.astype('float32')/255
test_x = test_x.astype('float32')/255

#print the shapes of the data arrays
print("Train Images: ",train_x.shape) #(60000,28,28)
print("Train Labels: ",train_y.shape)
print("Test Images: ",test_x.shape)
print("Test Labels: ",test_y.shape)

#Flatten the images
train_x = train_x.reshape(60000,784)
test_x = test_x.reshape(10000,784)

#Encode the labels to vectors
train_y = keras.utils.to_categorical(train_y,10)
test_y = keras.utils.to_categorical(test_y,10)

#Defining the model
model = Sequential() #using keras model API
model.add(Dense(units=128,activation="relu",input_shape=(784,))) #relu classifier is max(0,Z) where Z is the component
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=10,activation="softmax"))#softmax classifer/activation,aka sigmoid function, is 1 / (1 + (e^-z)) where z is component

#Specifying the training components
model.compile(optimizer=SGD(0.01),loss="categorical_crossentropy",metrics=["accuracy"])
#Fit the model
model.fit(train_x,train_y,batch_size=32,epochs=10,shuffle=True,verbose
=1)

#Evaluate the accuracy of the test dataset
accuracy = model.evaluate(x=test_x,y=test_y,batch_size=32)
print("Accuracy: ",accuracy[1])
