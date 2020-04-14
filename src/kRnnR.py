import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

BATCH_START = 0
TIME_STEPS = 20 #一次input分为20步
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.003

#用sin去预测cos
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START =BATCH_START + TIME_STEPS
    #plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    #plt.show()
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

model=Sequential()

#LSTM
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE,TIME_STEPS,INPUT_SIZE),
    units=CELL_SIZE,
    return_sequences=True,#在每一个time step之后，要不要输出结果；默认是不要，这里需要
    stateful=True,#batch和batch之间有没有联系
))

#add output layer; TieDistributed表示对LSTM每一个step的结果进行output
model.add(TimeDistributed(Dense(units=OUTPUT_SIZE)))

adam = Adam(lr=LR)

model.compile(optimizer=adam,loss='mse')

for step in range(501):
    X_batch,Y_batch,xs=get_batch()
    cost=model.train_on_batch(X_batch,Y_batch)
    pred=model.predict(X_batch,BATCH_SIZE)

    if step % 10 == 0:

        plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.1)

        print('train cost: ', cost)
