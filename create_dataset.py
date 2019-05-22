# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import signal
import pickle

def rate_to_sma(csv_file_date, s_window, m_window, l_window): #週間データを読み取ってきて移動平均線を3種類作る関数(実際のところ,今回は長期の移動平均線sma_lしか使っていない)
    rate = pd.read_csv('hist_datas/GBPJPY/hist_week(' + csv_file_date + ').csv',
                               header=None, names=["rate"], index_col=0)
    
    sma_s = rate.rolling(window=s_window).mean().rename(columns={'rate': 'sma_s'})
    sma_m = rate.rolling(window=m_window).mean().rename(columns={'rate': 'sma_m'})
    sma_l = rate.rolling(window=l_window).mean().rename(columns={'rate': 'sma_l'})
    
    rate_list = pd.concat([rate, sma_s, sma_m, sma_l], axis=1)
    
    return rate_list


def make_dataset(rate_list, dataset, train_number, predict_number): #長期の移動平均線の極小値or極大値+30秒(6点*5秒)までがトレーニングデータ(360点, 360*5秒=30分), その先60点*5秒=5分　後のデータも一緒に保存
    maxid = signal.argrelmax(rate_list['sma_l'].as_matrix())
    minid = signal.argrelmin(rate_list['sma_l'].as_matrix())
    for i in range(len(maxid[0])):
        if maxid[0][i] > train_number and maxid[0][i] < len(rate_list['rate'])-1-predict_number-6:
            dataset = np.append(dataset, [rate_list['rate'][maxid[0][i]+1-train_number+6: maxid[0][i]+1+predict_number+6]],axis=0)
    for i in range(len(minid[0])):
        if minid[0][i] > train_number and minid[0][i] < len(rate_list['rate'])-1-predict_number-6:
            dataset = np.append(dataset, [rate_list['rate'][minid[0][i]+1-train_number+6: minid[0][i]+1+predict_number+6]],axis=0)
            
    return dataset
        

#↓OneDriveのデータと対応している
curr = 'GBP'
year = np.array([2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018,
                 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019])
month = np.array([10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5])
day = np.array([6, 11, 13, 16, 20, 23, 27, 3, 10, 17, 22, 23, 24, 1, 13, 14, 15, 22, 25, 29, 3, 5, 12, 17, 20, 27, 2, 8, 14, 16, 2, 9, 12, 16, 23, 30, 6, 13, 20, 27, 4, 8])

train_number = 360
predict_number = 60
dataset = np.zeros((2, train_number+predict_number), dtype=float)

s_window = 12
m_window = 30
l_window = 120

for i in range(len(year)):
    print(i)
    csv_file_date = str(year[i]) + '-' + str(month[i]) + '-' + str(day[i])
    
    rate_list = rate_to_sma(csv_file_date, s_window, m_window, l_window)
    dataset = make_dataset(rate_list, dataset, train_number, predict_number)
    
dataset = dataset[2:]
path = 'dataset/GBPJPY/data_360_60_6.pickle'
with open(path, 'wb') as f:
    pickle.dump(dataset, f)