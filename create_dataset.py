# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import signal
import pickle
import matplotlib.pyplot as plt



class Create_dataset:
    def __init__(self, curr, csv_file_dates, train_number, predict_number, delay_number, window, split):
        self.curr = curr
        self.csv_file_dates = csv_file_dates
        self.train_number = train_number
        self.predict_number = predict_number
        self.delay_number = delay_number
        self.window = window
        self.split = split
        if self.split==True:
            self.dataset_max = np.zeros((2, self.train_number+self.predict_number), dtype=float)
            self.dataset_min = np.zeros((2, self.train_number+self.predict_number), dtype=float)
        else:
            self.dataset = np.zeros((2, self.train_number+self.predict_number), dtype=float)
        
    
    def _rate_to_sma(self): #週間データを読み取ってきて移動平均線を追加する関数
        rate = pd.read_csv('hist_data/' + self.curr + 'JPY/' + str(self.csv_file_date) + '.csv',
                           header=None, names=["rate"], index_col=0)
        sma = rate.rolling(window=self.window).mean().rename(columns={'rate': 'sma'})
        self.rate_list = pd.concat([rate, sma], axis=1)
        
        return self.rate_list
    
    
    def _make_data(self): #長期の移動平均線の極小値or極大値+30秒(6点*5秒)までがトレーニングデータ(360点, 360*5秒=30分), その先60点*5秒=5分　後のデータも一緒に保存
        maxid = signal.argrelmax(self.rate_list['sma'].as_matrix()) #移動平均線が極大値を取るときのインデックス
        minid = signal.argrelmin(self.rate_list['sma'].as_matrix()) #移動平均線が極小値を取るときのインデックス
        for i in range(len(maxid[0])):
            if maxid[0][i] > self.train_number and maxid[0][i] < len(self.rate_list['rate'])-1-self.predict_number-self.delay_number:
                self.dataset = np.append(self.dataset,
                                         [self.rate_list['rate'][maxid[0][i]-self.train_number+self.delay_number: maxid[0][i]+self.predict_number+self.delay_number]],
                                         axis=0)
        for i in range(len(minid[0])):
            if minid[0][i] > self.train_number and minid[0][i] < len(self.rate_list['rate'])-1-self.predict_number-self.delay_number:
                self.dataset = np.append(self.dataset,
                                         [self.rate_list['rate'][minid[0][i]-self.train_number+self.delay_number: minid[0][i]+self.predict_number+self.delay_number]],
                                         axis=0)
                
        return self.dataset
    
    
    def _make_data_split(self): #極大値のデータと極小値のデータを分ける
        maxid = signal.argrelmax(self.rate_list['sma'].as_matrix()) #移動平均線が極大値を取るときのインデックス
        minid = signal.argrelmin(self.rate_list['sma'].as_matrix()) #移動平均線が極小値を取るときのインデックス
        for i in range(len(maxid[0])):
            if maxid[0][i] > self.train_number and maxid[0][i] < len(self.rate_list['rate'])-1-self.predict_number-self.delay_number:
                self.dataset_max = np.append(self.dataset_max,
                                             [self.rate_list['rate'][maxid[0][i]-self.train_number+self.delay_number: maxid[0][i]+self.predict_number+self.delay_number]],
                                             axis=0)
        for i in range(len(minid[0])):
            if minid[0][i] > self.train_number and minid[0][i] < len(self.rate_list['rate'])-1-self.predict_number-self.delay_number:
                self.dataset_min = np.append(self.dataset_min,
                                             [self.rate_list['rate'][minid[0][i]-self.train_number+self.delay_number: minid[0][i]+self.predict_number+self.delay_number]],
                                             axis=0)
                
        return self.dataset_max, self.dataset_min
            
    
    def make_dataset(self):
        for i in range(len(self.csv_file_dates)):
            print(i)
            self.csv_file_date = self.csv_file_dates[i]
            self._rate_to_sma()
            if self.split==True:
                self._make_data_split()
            else:
                self._make_data()
                
        if self.split==True:
            self.dataset_max = self.dataset_max[2:]
            self.dataset_min = self.dataset_min[2:]
        else:
            self.dataset = self.dataset[2:]
            
        if self.split==True:
            self.dataset = {}
            self.dataset['max'] = self.dataset_max
            self.dataset['min'] = self.dataset_min
        
        return self.dataset
    
    
    def save_dataset(self):
        if self.split==True:
            path = 'dataset/' + self.curr + 'JPY/data_' + str(self.train_number) + '_' + str(self.predict_number) + '_' + str(self.delay_number) + '_' + str(self.window) +'_split.pickle'
        else:
            path = 'dataset/' + self.curr + 'JPY/data_' + str(self.train_number) + '_' + str(self.predict_number) + '_' + str(self.delay_number) + '_' + str(self.window) +'.pickle'
        with open(path, 'wb') as f:
            pickle.dump(self.dataset, f)
            
    
    def display(self, data):
        i = np.arange(len(data))
        maxid = signal.argrelmax(data)
        minid = signal.argrelmin(data)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        lines0, = ax.plot(i, data)
        lines1, = ax.plot(i[maxid], data[maxid], 'o')
        lines1, = ax.plot(i[minid], data[minid], 'o')
        #ax.set_xlim((-lim, lim))
        #ax.set_ylim((-lim, lim))
        plt.show()
        


if __name__ == '__main__':
    #OneDriveのデータと対応している
    curr = 'GBP'
    csv_file_dates = np.array([20181006, 20181011, 20181013, 20181016, 20181020, 20181023, 20181027, 20181103, 20181110, 20181117,
                              20181122, 20181123, 20181124, 20181201, 20181213, 20181214, 20181215, 20181222, 20181225, 20181229,
                              20190103, 20190105, 20190112, 20190117, 20190120, 20190127, 20190202, 20190208, 20190214, 20190216,
                              20190302, 20190309, 20190312, 20190316, 20190323, 20190330, 20190406, 20190413, 20190420, 20190427,
                              20190504, 20190508, 20190513, 20190518])
    
    #ハイパーパラメータ
    train_number = 120 #トレーニングデータの長さ
    predict_number = 6 #予測の長さ 6→30秒
    delay_number = 2 #極値の何秒後に予測を行うか 3→15秒
    window = 120 #移動平均線の幅
    
    C = Create_dataset(curr, csv_file_dates, train_number, predict_number, delay_number, window, split=False)
    C.make_dataset()
    C.save_dataset()
    
    """
    #動作チェック
    C.csv_file_date = csv_file_dates[0]
    C._rate_to_sma()
    #C._make_data_split()
    #dataset_max = C.dataset_max
    #dataset_min = C.dataset_min
    maxid = signal.argrelmax(C.rate_list['sma'].as_matrix())
    C.dataset_max = np.append(C.dataset_max, [C.rate_list['sma'][maxid[0][10]-120+2: maxid[0][10]+6+2]],axis=0)
    C.display(C.dataset_max[2])
    #C.display(dataset_min[2])
    """