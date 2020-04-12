import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

#使用keras下载数据集有点问题，手动下载数据集
mnist = np.load("../dataset/mnist.npz")
X_train, y_train, X_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']

#data pre-processing
X_train = X_train.reshape(-1,1,28,28)/255 #第二个1表示只有一个channel，黑白照片
X_test = X_test.reshape(-1,1,28,28)/255

#使用keras编写的np_utils，把标签分为10类，每一类都用10位二进制表示，某一位为1其他为0
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

model=Sequential()

#卷积层
model.add(Convolution2D(
    filters=32,#滤波器的个数;每个滤波器会生成一个feature的图片，之后的图片会是32层
    kernel_size=5,#滤波器的长宽都为5个像素
    padding='same',#padding的模式
    batch_input_shape=(None,1,28,28),#输入的大小
    strides=1,#过滤器移动的步长
))

model.add(Activation('relu'))

#pooling
model.add(MaxPooling2D(
    pool_size=(2,2),#池化器的大小
    strides=2,#池化器移动的步长
    padding='same'
))

#Cov
model.add(Convolution2D(64,5,padding='same',strides=1))

model.add(Activation('relu'))

#pooling
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))

#fully connected
model.add(Flatten())#抹平为一维的数据
model.add(Dense(1024))#全连接层
model.add(Activation('relu'))

#fully connected
model.add(Dense(10))
model.add(Activation('softmax'))

adam=Adam(lr=1e-4)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

print('Training---------')
model.fit(X_train,y_train,batch_size=32,epochs=2)

print('Testing---------')
loss,accuracy=model.evaluate(X_test,y_test)

print('loss: ',loss)
print('accuracy: ',accuracy)