# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, BatchNormalization, Activation, Dense, concatenate, Input
from keras import optimizers
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import numpy as np
import pickle



class Neuralnet_class:
    def __init__(self, curr, train_number, predict_number, delay_number, window, test_number, split):
        self.curr = curr
        self.train_number = train_number
        self.predict_number = predict_number
        self.delay_number = delay_number
        self.window = window
        self.test_number = test_number
        self.split = split
    

    def load_dataset(self):    
        if self.split==True:
            path = 'dataset/' + curr + 'JPY/data_' + str(self.train_number) + '_' + str(self.predict_number) + '_' + str(self.delay_number) + '_' + str(self.window) + '_split.pickle'
        else:
            path = 'dataset/' + curr + 'JPY/data_' + str(self.train_number) + '_' + str(self.predict_number) + '_' + str(self.delay_number) + '_' + str(self.window) + '.pickle'
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        if self.split==True:
            self.data = self.data['max']
            
        return self.data
    
    
    def training_dataset_generation(self): #トレーニングデータとトレーニングラベル生成
        self.x_data = self.data[:, 0: self.train_number]
        self.t_data = np.zeros((len(self.x_data), 2), dtype=int)
        for i in range(len(self.x_data)):
            if self.data[i, self.train_number-1] < self.data[i, self.train_number+self.predict_number-1]:
                self.t_data[i] = np.array([1, 0])
            else:
                self.t_data[i] = np.array([0, 1])
        
        #各データ毎に標準化
        for i in range(len(self.x_data)):
            mean = np.mean(self.x_data[i])
            std = np.std(self.x_data[i]) + 1e-7
            self.x_data[i] = (self.x_data[i] - mean) / std
            
            
    def test_dataset_generation_randam(self): #トレーニングデータとテストデータをランダムに分離
        mask = np.random.choice(len(self.x_data), self.test_number, replace=False)
        ind = np.ones(len(self.x_data), dtype=bool)
        ind[mask] = False
        self.x_test = self.x_data[mask]
        self.x_train = self.x_data[ind]
        self.t_test = self.t_data[mask]
        self.t_train = self.t_data[ind]
        
    
    def test_dataset_generation_latest(self): #最後をテストデータに分離
        self.x_test = self.x_data[len(self.x_data)-self.test_number: len(self.x_data)]
        self.x_train = self.x_data[0: len(self.x_data)-self.test_number]
        self.t_test = self.t_data[len(self.x_data)-self.test_number: len(self.x_data)]
        self.t_train = self.t_data[0: len(self.x_data)-self.test_number]


    def model_and_learning(self): #Kerasモデルの構築と学習
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=self.train_number, activation="relu"))
        self.model.add(Dropout(0.6))
        self.model.add(Dense(2, activation="softmax"))
        self.model.summary()
        self.adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        self.tbcb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True)
        self.history = self.model.fit(self.x_train, self.t_train, epochs=30, batch_size=100, verbose=2, validation_split=0.1, callbacks=[self.tbcb])


    def save_model(self):
        if self.split==True:
            model_json = self.model.to_json()
            with open('params/' + curr + 'JPY/data_' + str(self.train_number) + '_' + str(self.predict_number) + '_' + str(self.delay_number) + '_' + str(self.window) + '_split_model.json', mode='w') as f:
                f.write(model_json)
            self.model.save_weights('params/' + curr + 'JPY/data_' + str(self.train_number) + '_' + str(self.predict_number) + '_' + str(self.delay_number) + '_' + str(self.window) + '_split_weights.h5')
            with open('params/' + curr + 'JPY/data_' + str(self.train_number) + '_' + str(self.predict_number) + '_' + str(self.delay_number) + '_' + str(self.window) + '_split_history.pickle', mode='wb') as f:
                pickle.dump(self.history.history, f)
        else:
            model_json = self.model.to_json()
            with open('params/' + curr + 'JPY/data_' + str(self.train_number) + '_' + str(self.predict_number) + '_' + str(self.delay_number) + '_' + str(self.window) + '_model.json', mode='w') as f:
                f.write(model_json)
            self.model.save_weights('params/' + curr + 'JPY/data_' + str(self.train_number) + '_' + str(self.predict_number) + '_' + str(self.delay_number) + '_' + str(self.window) + '_weights.h5')
            with open('params/' + curr + 'JPY/data_' + str(self.train_number) + '_' + str(self.predict_number) + '_' + str(self.delay_number) + '_' + str(self.window) + '_history.pickle', mode='wb') as f:
                pickle.dump(self.history.history, f)


    def predict(self): #テストデータに対する予測
        self.p = self.model.predict(self.x_test)
        
        return self.p


    def simple_accuracy(self):
        result = np.zeros(2)
        for i in range(len(self.p)):
            result[1] += 1
            if (self.p[i, 0] > self.p[i, 1] and self.t_test[i, 0] > self.t_test[i, 1]) or (self.p[i, 0] < self.p[i, 1] and self.t_test[i, 0] < self.t_test[i, 1]):
                result[0] += 1
        print(str(result[0]/result[1]*100) + '%(' + str(result[0]) + '/' + str(result[1]) + ')')
    
    
    def threshold_accuracy(self, th):
        result = np.zeros(2)
        for i in range(len(self.p)):
            if abs(self.p[i, 0]-self.p[i, 1]) > th:
                result[1] += 1
                if (self.p[i, 0] > self.p[i, 1] and self.t_test[i, 0] > self.t_test[i, 1]) or (self.p[i, 0] < self.p[i, 1] and self.t_test[i, 0] < self.t_test[i, 1]):
                    result[0] += 1
        print(str(result[0]/result[1]*100) + '%(' + str(result[0]) + '/' + str(result[1]) + ')')



if __name__ == '__main__':
    curr = 'GBP'
    train_number = 60 #トレーニングデータの長さ
    predict_number = 6 #予測の長さ 6→30秒
    delay_number = 3 #極値の何秒後に予測を行うか 3→15秒
    window = 70 #移動平均線の幅
    
    NN = Neuralnet_class(curr, train_number, predict_number, delay_number, window, 10000, split=False)
    NN.load_dataset()
    NN.training_dataset_generation()
    NN.test_dataset_generation_latest()
    NN.model_and_learning()
    NN.save_model()
    NN.predict()
    NN.simple_accuracy()
    NN.threshold_accuracy(0.08)