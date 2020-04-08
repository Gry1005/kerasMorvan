import numpy as np
#Sequential表示按顺序建立的神经网络
from keras.models import Sequential
#dense是全连接层
from keras.layers import Dense
import matplotlib.pyplot as plt

np.random.seed(1337) #设置随机数种子

#create data
#在-1,1之间均匀地生成200个数
X=np.linspace(-1,1,200)
np.random.shuffle(X)    # randomize the data
#normal表示正态分布，loc, scale, size；loc表示正态分布的中心点，scale是标准差，size是输出多少个值
Y=0.5*X+2+np.random.normal(0,0.05,200)

plt.scatter(X,Y)
plt.show()

X_train, Y_train = X[:160], Y[:160] #训练集是前160个点
X_test, Y_test=X[160:], Y[160:] #测试集是后40个点

#build network
#建立一个按顺序的神经网络模型
model = Sequential()

#加入神经元层
model.add(Dense(units=1,input_dim=1)) #输入是x,输出是y; units表示这一层有几个神经元（即输出的规模），input_dim表示输入的规模

#设置误差函数和优化器
model.compile(loss='mse',optimizer='sgd')

#training
print("Training-----")
#训练300代
for step in range(301):
    #模型训练;
    cost=model.train_on_batch(X_train,Y_train)

    if step%50==0:
        print('training cost: ',cost)

#test
print("Testing-----")
cost=model.evaluate(X_test,Y_test,batch_size=40)
print('Test cost: ',cost)
W,b=model.layers[0].get_weights()
print('Weights: ',W,' b: ',b)

#ploting
Y_pred=model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()









