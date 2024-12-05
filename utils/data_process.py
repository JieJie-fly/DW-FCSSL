import numpy as np
import math
import pandas as pd
import torch
import os

def error_anlysis(real, predict):
    error = []
    squaredError = []
    absError = []
    for i in range(len(real)):
        error.append(real[i]-predict[i])
        squaredError.append(error[i]**2)
        absError.append(abs(error[i]))
    a = sum(squaredError) / len(squaredError)   # 均方误差MSE
    b = math.sqrt(sum(squaredError) / len(squaredError))  # 均方根误差RMSE
    c = sum(absError) / len(absError)   # 平均绝对误差MAE
    return a, b, c

def get_dataset(path):
    origin_data = np.load(path)
    label = origin_data[:, -1]
    data = np.delete(origin_data, -1, 1)
    label = torch.from_numpy(label)
    label = label.float()
    data = torch.from_numpy(data)
    data = data.float()
    return data, label

def time_window(data, win_size):
    X = []
    Y = []
    for i  in range(len(data)-win_size):
        temp_x = data[i:i + win_size, :-1]
        temp_y = data[i + win_size, -1]
        X.append(temp_x)
        Y.append(temp_y)
    X = np.array(X)
    Y = np.array(Y)

    # X (data.shape[0]-win_size, win_size, feature_num)
    # Y (data.shape[0]-win_size, 1)

    return X, Y


# 读取数据并对数据进行时间窗口处理并且将每个数据保存为csv文件
def data_timewindow_save(win_size=10):
    path = os.getcwd()
    Path_load = path + '/data/raw_data'
    Path_dataset_save = path + '/data/raw_data_timewidow(10)/c3(13,14,15,16)/'
    Path_label_save = path + '/data/raw_label_timewidow(10)/c3(13,14,15,16)/'

    for i in range(13, 17):
        data_i, label_i = get_dataset(Path_load + '/#' + str(i) + '_normalized.npy')
        label_i = label_i.reshape(label_i.shape[0], 1)
        data_i = np.concatenate((data_i, label_i), axis=1)

        for tw in range(len(data_i)-win_size):
            temp_x = data_i[tw:tw + win_size, :-1]
            temp_y = data_i[tw + win_size, -1]
            temp_y = temp_y.reshape(temp_y.shape)

            temp_x = pd.DataFrame(temp_x)
            temp_y = pd.DataFrame([temp_y])

            temp_x.to_csv(Path_dataset_save + str(i) + '_' + str(tw) + '.csv')
            temp_y.to_csv(Path_label_save + str(i) + '_' + str(tw) + '.csv')
