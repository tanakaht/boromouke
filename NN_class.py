# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, BatchNormalization, Activation, Dense, concatenate, Input
from keras import optimizers
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import numpy as np
import pickle

curr = 'GBP'
train_shape = 360
pre_shape = 60



path = 'dataset/' + curr + 'JPY/data_360_60_6.pickle'
with open(path, 'rb') as f:
    data = pickle.load(f)

#トレーニングデータとトレーニングラベル生成
x_data0 = data[:, 0: train_shape]
t_data0 = np.zeros((len(x_data0), 2), dtype=int)
for i in range(len(x_data0)):
    if data[i, train_shape-1] < data[i, train_shape+pre_shape-1]:
        t_data0[i] = np.array([1, 0])
    else:
        t_data0[i] = np.array([0, 1])

#各データ毎に標準化
for i in range(len(x_data0)):
    mean = np.mean(x_data0[i])
    std = np.std(x_data0[i]) + 1e-7
    x_data0[i] = (x_data0[i] - mean) / std

#トレーニングデータとテストデータ(10000)の分離
mask = np.random.choice(len(x_data0), 10000, replace=False)
ind = np.ones(len(x_data0), dtype=bool)
ind[mask] = False
x_test = x_data0[mask]
x_data = x_data0[ind]
t_test = t_data0[mask]
t_data = t_data0[ind]


#x_data = x_data.reshape(len(x_data), train_shape, 1) #CNNのときいる

#Kerasモデルの構成と学習
model = Sequential()
model.add(Dense(100, input_dim=train_shape, activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(2, activation="softmax"))
model.summary()
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
tbcb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True)
history = model.fit(x_data, t_data, epochs=30, batch_size=100, verbose=2, validation_split=0.1, callbacks=[tbcb])

#テストデータに対する予測
p = model.predict(x_test)

#正答率チェック
result0 = np.zeros(2)
result1 = np.zeros(2)
for i in range(len(p)):
    result0[1] += 1
    if (p[i, 0] > p[i, 1] and t_test[i, 0] > t_test[i, 1]) or (p[i, 0] < p[i, 1] and t_test[i, 0] < t_test[i, 1]):
        result0[0] += 1
    if abs(p[i, 0]-p[i, 1]) > 0.08:
        result1[1] += 1
        if (p[i, 0] > p[i, 1] and t_test[i, 0] > t_test[i, 1]) or (p[i, 0] < p[i, 1] and t_test[i, 0] < t_test[i, 1]):
            result1[0] += 1
#テストデータ10000全部における正答率
print(result0[0]/result0[1])
#テストデータ10000の内出力の差の絶対値が0.08以上のときの正答率
print(result1[0]/result1[1])