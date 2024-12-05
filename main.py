import os
import pandas as pd
import torch
from utils.data_aug import *
from utils.federated_learning import *
from utils.data_process import *
from utils.result_diaplay import figure_visualization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = os.getcwd()

# 保存增强数据
def load_data(data_dir):
    files = list(Path(data_dir).rglob("*.csv"))
    Freq = []
    Crop = []
    Mask_freq = []
    Mask_crop = []
    Mask_raw = []
    Raw = []

    for i in range(len(files)):
        aug = Aug(data_dir, 0.5, 1, 0.8, 3, 3)
        data = aug.__getitem__(i)
        Freq.append(data[0])
        Crop.append(data[1])
        Mask_freq.append(data[2])
        Mask_crop.append(data[3])
        Raw.append(data[4])
    return Freq, Crop, Mask_freq, Mask_crop, Raw


def load_label(label_dir):
    files = list(Path(label_dir).rglob("*.csv"))
    Label = []
    for i in range(len(files)):
        file_i = str(files[i])
        label = pd.read_csv(file_i)
        label = np.array(label)[:, 1:]
        Label.append(label)
    return Label


# 加载三个客户端的标签
Label_c1 = load_label(path+'/data/raw_label_timewidow(10)/RW1_2_7_8/')
Label_c2 = load_label(path+'/data/raw_label_timewidow(10)/RW3_4_5_6/')
Label_c3 = load_label(path+'/data/raw_label_timewidow(10)/RW9_10_11_12/')


# 加载每个客户端的增强数据以及原始数据
Freq1, Crop1, Mask_freq1, Mask_crop1, Raw1 = load_data(path+'/data/raw_data_timewidow(10)/RW1_2_7_8/')
Freq2, Crop2, Mask_freq2, Mask_crop2, Raw2 = load_data(path+'/data/raw_data_timewidow(10)/RW3_4_5_6/')
Freq3, Crop3, Mask_freq3, Mask_crop3, Raw3 = load_data(path+'/data/raw_data_timewidow(10)/RW9_10_11_12/')


# 将增强数据加载到一起
dataset1 = CustomDataset(Freq1, Crop1, Mask_freq1, Mask_crop1)
dataset2 = CustomDataset(Freq2, Crop2, Mask_freq2, Mask_crop2)
dataset3 = CustomDataset(Freq3, Crop3, Mask_freq3, Mask_crop3)

data = {}
data[0] = dataset1
data[1] = dataset2
data[2] = dataset3

num_clients = 3
rounds = 20
num_epochs = 10


if __name__ == '__main__':
    server = federated_learning(num_clients, rounds, num_epochs, data)
    figure_visualization()
    pass


