#函数式写法
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Input
from keras.optimizers import Adam

#使用keras下载数据集有点问题，手动下载数据集
mnist = np.load("../dataset/mnist.npz")
X_train, y_train, X_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']

#data pre-processing
X_train = X_train.reshape(-1,28,28,1)/255 #最后一个1表示只有一个channel，黑白照片
X_test = X_test.reshape(-1,28,28,1)/255

#使用keras编写的np_utils，把标签分为10类，每一类都用10位二进制表示，某一位为1其他为0
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

inputs=Input(batch_shape=(None,28,28,1))
x=Convolution2D(filters=32,kernel_size=5,padding='same',strides=1,activation='relu')(inputs)
x=MaxPooling2D(pool_size=(2,2),strides=2,padding='same')(x)
x=Convolution2D(64,5,padding='same',strides=1,activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2),strides=2,padding='same')(x)
x=Flatten()(x)
x=Dense(1024,activation='relu')(x)
predictions=Dense(10,activation='softmax')(x)

model=Model(inputs=inputs,outputs=predictions)

adam=Adam(lr=1e-4)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

print('Training---------')
model.fit(X_train,y_train,batch_size=32,epochs=2)

print('Testing---------')
loss,accuracy=model.evaluate(X_test,y_test)

print('loss: ',loss)
print('accuracy: ',accuracy)