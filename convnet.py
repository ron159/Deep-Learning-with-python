from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models

(trian_images,trian_labels),(test_images,test_labels)=mnist.load_data()
trian_images=trian_images.reshape((60000,28,28,1))
trian_images=trian_images.astype('float32')/255

test_images=test_images.reshape((10000,28,28,1))
test_images=test_images.astype('float32')/255

trian_labels=to_categorical(trian_images)
test_labels=to_categorical(trian_labels)

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten()) 
model.add(layers.Dense(64, activation='relu')) 
model.add(layers.Dense(10, activation='softmax')) 

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(test_images,trian_labels,epochs=5,batch_size=64)