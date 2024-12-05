# -*-coding:utf-8-*-
import torch
from utils.model import Encoder
import torch.nn as nn

# 定义Server
# Server需要实现的功能：
# 1.接收来自源域Client的特征向量和模型参数
# 2.根据每个源域的权重对源域模型进行加权聚合

class Server(object):
    def __init__(self):
        self.global_model = Encoder(4, 4, 64)

    def get_global_model(self):
        return self.global_model

    def aggregate(self, client_params, weight):
        # 计算平均参数
        avg_params = {}

        for param_name in client_params[0].keys():
            avg_params[param_name] = torch.stack([weight * params[param_name] for params, weight in zip(client_params, weight)]).sum(dim=0)

        # 更新全局模型参数
        self.global_model.load_state_dict(avg_params)

    def get_global_params(self):
        return self.global_model.state_dict()

    ## server模型获取目标域的模型参数
    def receive_target_model(self, target_model_params):
        self.global_model.load_state_dict(target_model_params)

    def predict(self, data):
        self.global_model.eval()
        with torch.no_grad():
            fea, outputs = self.global_model(data)
        return fea, outputs.squeeze().numpy()

    def evaluate(self, data, target):
        self.global_model.eval()
        with torch.no_grad():
            fea, outputs = self.global_model(data)
            loss = nn.MSELoss()(outputs.squeeze(), target)
        return loss.item()

    def save_model(self):
        return self.global_model.state_dict()