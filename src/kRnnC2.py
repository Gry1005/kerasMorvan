#使用函数式

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, SimpleRNN, Activation, Dense
from keras.optimizers import Adam

TIME_STEPS = 28     # same as the height of the image;对一个数据，读取多少步
INPUT_SIZE = 28     # same as the width of the image；每一步读取多少数据
BATCH_SIZE = 50
BATCH_INDEX = 0     #用来生成数据
OUTPUT_SIZE = 10
CELL_SIZE = 50      #hidden unit的数量
LR = 0.001

#使用keras下载数据集有点问题，手动下载数据集
mnist = np.load("../dataset/mnist.npz")
X_train, y_train, X_test, y_test = mnist['x_train'], mnist['y_train'], mnist['x_test'], mnist['y_test']

#data pre-processing; 注意和CNN的处理不一样，因为这一次是处理为序列化数据
X_train = X_train.reshape(-1,28,28)/255 #第二个1表示只有一个channel，黑白照片
X_test = X_test.reshape(-1,28,28)/255

#使用keras编写的np_utils，把标签分为10类，每一类都用10位二进制表示，某一位为1其他为0
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

inputs = Input(batch_shape=(None,TIME_STEPS,INPUT_SIZE))
x=SimpleRNN(units=CELL_SIZE)(inputs)
predictions=Dense(units=OUTPUT_SIZE,activation='softmax')(x)

model = Model(inputs=inputs,outputs=predictions)

#优化器和model编译
adam=Adam(lr=LR)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

#另一种训练方法
'''
for step in range(4001):
    #从训练集中分批截取数据
    X_batch=X_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE,:,:]
    Y_batch=y_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE,:]
    cost=model.train_on_batch(X_batch,Y_batch)

    BATCH_INDEX=BATCH_INDEX+BATCH_SIZE
    BATCH_INDEX=0 if BATCH_INDEX>=X_train.shape[0] else BATCH_INDEX

    if step%500==0:
        cost,accuracy=model.evaluate(X_test,y_test,batch_size=y_test.shape[0],verbose=False)
        print('Test cost: ',cost, 'accuracy: ',accuracy)
'''

print('Training---------')
model.fit(X_train,y_train,batch_size=50,epochs=3)

print('Testing---------')
loss,accuracy=model.evaluate(X_test,y_test)

print('loss: ',loss)
print('accuracy: ',accuracy)
