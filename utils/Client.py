# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
from utils.model import Encoder
import torch.utils.data as Data
from utils.LOSS import *
from tqdm import tqdm
import time
import logging
import matplotlib.pyplot as plt
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


Batch_size = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

class Client(object):

    def __init__(self, dataset):
        # self.data = torch.tensor(data, dtype=torch.float32)
        # self.target = torch.tensor(target, dtype=torch.float32)
        self.data = dataset
        self.model = Encoder(4, 4, 64)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)

    def train(self, num_epochs, global_model):
        self.model.training
        self.model.to(device)
        global_model.eval()
        global_model.to(device)
        train_dataloader = Data.DataLoader(dataset=self.data, batch_size=Batch_size, shuffle=True, drop_last=True)
        # start_time = time.time()
        Loss = []
        Loss_contra = []
        Loss_recon = []

        pbar = tqdm(range(num_epochs), desc='Training Progress')
        pbar.ncols = 80

        for epoch in pbar:
            for batch in train_dataloader:
                freq_batch, crop_batch, mask_freq_batch, mask_crop_batch = batch
                self.optimizer.zero_grad()

                freq_batch = freq_batch.clone().detach().to(device)
                mask_freq_batch = mask_freq_batch.clone().detach().to(device)
                crop_batch = crop_batch.clone().detach().to(device)
                mask_crop_batch = mask_crop_batch.clone().detach().to(device)

                freq_batch = torch.squeeze(freq_batch)
                mask_freq_batch = torch.squeeze(mask_freq_batch)
                crop_batch = torch.squeeze(crop_batch)
                mask_crop_batch = torch.squeeze(mask_crop_batch)

                freq_batch = freq_batch.clone().detach().float()
                mask_freq_batch = mask_freq_batch.clone().detach().float()
                crop_batch = crop_batch.clone().detach().float()
                mask_crop_batch = mask_crop_batch.clone().detach().float()

                out1, inter1 = self.model(freq_batch, mask_freq_batch)
                out2, inter2 = self.model(crop_batch, mask_crop_batch)

                out3, _ = global_model(crop_batch, freq_batch)

                # calculate contrastive loss
                loss_contra_local = 0.01 * hierarchical_contrastive_loss(out1, out2)
                loss_contra_global = 0.01 * hierarchical_contrastive_loss(out1, out3)
                loss_contra = (loss_contra_local + loss_contra_global) / 2

                # calculate reconstructive loss
                loss_recon1 = criterion(inter1, mask_freq_batch)
                loss_recon2 = criterion(inter2, mask_crop_batch)
                loss_recon = (loss_recon1 + loss_recon2) / 2

                loss = loss_recon + 0.1 * loss_contra
                loss.backward()
                self.optimizer.step()

                Loss.append(loss)
                Loss_contra.append(loss_contra)
                Loss_recon.append(loss_recon)

            epoch_total_loss = sum(Loss) / len(Loss)
            epoch_contra_loss = sum(Loss_contra) / len(Loss_contra)
            epoch_recon_loss = sum(Loss_recon) / len(Loss_recon)

            # print("Epoch: %d Total_Loss: %f Contra_loss: %f Recon_loss: %f" % \
            #             (epoch, epoch_total_loss, epoch_contra_loss, epoch_recon_loss))

        return epoch_total_loss

    # 利用训练完成的源域模型输出源域数据的特征
    def get_source_feature(self):
        self.model.eval()
        with torch.no_grad():
            fea, outputs = self.model(self.data)
        return fea

    # 利用训练完成的源域模型输出目标数据的特征
    def get_target_feature(self, tar_data):
        self.model.eval()
        with torch.no_grad():
            tar_fea, outputs = self.model(tar_data)
        return tar_fea

    # 接受Server模型参数
    def get_params(self):
        return self.model.state_dict()

    # 加载本地模型参数
    def set_params(self, params):
        self.model.load_state_dict(params)

    # # 将目标域模型参数传递给客户端
    # def transfer_to_client(self, target_params):
    #     self.model.set_params(target_params)

    # 对local模型进行评估，输出loss值
    def evaluate(self, data, target):
        self.model.eval()
        with torch.no_grad():
            fea, outputs = self.model(data)
            loss = nn.MSELoss()(outputs.squeeze(), target)
        return fea, loss.item()

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            fea, outputs = self.model(data)
        return outputs

    def save_model(self):
        return self.model

def data_typr_trans(data):
    data = torch.tensor([item.cpu().detach().numpy() for item in data])
    data = torch.squeeze(data)
    data = torch.tensor(data, dtype=torch.float)
    return data