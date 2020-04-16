#使用函数式写法

from keras.datasets import mnist
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input,Dense, Activation
from keras.optimizers import RMSprop

np.random.seed(1337)  # for reproducibility

#使用keras下载数据集有点问题，手动下载数据集
mnist = np.load("../dataset/mnist.npz")
X_train, y_train, X_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#data pre-processing
X_train = X_train.reshape(X_train.shape[0],-1)/255 #把(0,255)的像素点变为(0,1)
X_test = X_test.reshape(X_test.shape[0],-1)/255

#使用keras编写的np_utils，把标签分为10类，每一类都用10位二进制表示，某一位为1其他为0
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

inputs = Input(shape=(28*28,))
x=Dense(units=32,activation='relu')(inputs)
prediction=Dense(units=10,activation='softmax')(x)

model=Model(inputs=inputs,outputs=prediction)


#optimizer
rmsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

#模型编译
model.compile(
    optimizer=rmsprop,#默认的optimizer可以传入一个字符串
    loss='categorical_crossentropy', #crossentropy
    metrics=['accuracy'], #在训练的时候，计算一些值并输出
)

print('Training-------')
#使用fit，常用的
#nb_epoch表示训练多少大轮(一大轮是所有数据)
model.fit(X_train,y_train,epochs=2,batch_size=32)

print('Testing-------')
loss,accuracy=model.evaluate(X_test,y_test)

print('loss: ',loss)
print('accuracy: ',accuracy)






