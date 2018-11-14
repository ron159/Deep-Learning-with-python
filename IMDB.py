from keras.datesets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results
x_trian=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

y_trian=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])      
model.compile(optimizer=optimizers.RMSprop(1r=0.001),loss='binary_crossentropy',metrics=['accuracy'])   
model.compile(optimizer=optimizers.RMSprop(1r=0.001),loss='binary_crossentropy',metrics=[metrics.binary_accuracy]) 

x_val=x_trian[:10000]
partial_x_trian=x_trian[10000:] 

y_val=y_trian[:10000]
partial_y_trian=y_trian[10000:]

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(partial_x_trian,partial_y_trian,epochs=20,batch=512,validation_data=(x_val,y_val))
