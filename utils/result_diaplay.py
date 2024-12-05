import os
import pandas as pd
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_aug import *
from utils.federated_learning import *
from utils.model import Encoder,FCSSL
from utils.data_process import *
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

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


def data_type_trans(data):
    data = torch.tensor([item.cpu().detach().numpy() for item in data])
    data = torch.squeeze(data)
    data = torch.tensor(data, dtype=torch.float)
    return data

def label_type_trans(data):
    data = torch.tensor([item for item in data])
    data = torch.squeeze(data)
    data = torch.tensor(data, dtype=torch.float)
    return data

Raw_data1 = data_type_trans(Raw1[119:])
Raw_label1 = label_type_trans(Label_c1[119:])
Raw_data2 = data_type_trans(Raw2[34:62])
Raw_label2 = label_type_trans(Label_c2[34:62])
Raw_data3 = data_type_trans(Raw3[70:137])
Raw_label3 = label_type_trans(Label_c3[70:137])

def model_prediction(model, test_data):
    prediction = []
    for i in range(len(test_data)):
        te = test_data[i]
        te = te.reshape(1, te.shape[0], te.shape[1])
        pre, rep, _ = model(te, te)
        pre = [max(x, 0) for x in pre]
        pre = torch.asarray(pre)
        prediction.append(pre)
    prediction = torch.asarray(prediction)
    return prediction

def data_visualization(Raw_data, Raw_label, dict, result_dict,num):
    data = Raw_data
    label = Raw_label
    num_sample = int(len(data) * 0.3)
    test_data = data[num_sample:]
    model = FCSSL(4, 4, 64)
    model_dict1 = torch.load(dict)
    model.load_state_dict(model_dict1)
    model.eval()
    prediction = model_prediction(model, test_data)

    error = prediction-label[num_sample:]

    fig, ax1 = plt.subplots(figsize=(4, 5))

    # ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_facecolor('#F5F7F2')
    x_indices_pre = np.arange(num_sample, len(Raw_data))
    x_indices_label = np.arange(0, num_sample)
    ax1.set_xlabel('Cycle', fontsize=14, fontweight='bold',fontname='Times New Roman')
    ax1.tick_params(axis='y')
    ax1.tick_params(axis='x')
    ax1.set_ylabel('Normalized SOH', fontsize=14, fontweight='bold',fontname='Times New Roman')

    ax1.set_title(num, fontname='Times New Roman', fontsize=14, fontweight='bold')
    ax1.plot(x_indices_pre, prediction, '-^', c='#0000FF')
    ax1.plot(x_indices_label, label[:num_sample], '-^', c='#FFC000')
    ax1.plot(x_indices_pre, label[num_sample:], '-^', c='#E05DFF')

    ax2 = ax1.twinx()

    ax2.set_ylabel('Error', fontsize=14, fontweight='bold',fontname='Times New Roman')
    ax2.tick_params(axis='y')
    ax2.set_ylim([-0.2, 0.3])
    ax2.fill_between(x_indices_pre, error, color='lightgray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(result_dict, dpi=300)

def figure_visualization():
    data_visualization(Raw_data1, Raw_label1, 'model_saver/Client1/FCSSL_wt_30%_client1_#8.pt', 'result_visualization/#8.png','#8')
    data_visualization(Raw_data2, Raw_label2, 'model_saver/Client2/FCSSL_wt_30%_client2_#4.pt', 'result_visualization/#4.png','#4')
    data_visualization(Raw_data3, Raw_label3, 'model_saver/Client3/FCSSL_wt_30%_client3_#10.pt', 'result_visualization/#10.png','#10')

    image_paths = ['result_visualization/#10.png', 'result_visualization/#4.png', 'result_visualization/#8.png']
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.1, hspace=0.05)
    for gsp, image_path in zip(gs, image_paths):
        ax = fig.add_subplot(gsp)
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.axis('off')
    # 显示图表
    plt.show()

