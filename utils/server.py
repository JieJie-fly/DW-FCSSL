# -*-coding:utf-8-*-
import torch
from utils.model import Encoder
import torch.nn as nn



class Server(object):
    def __init__(self):
        self.global_model = Encoder(4, 4, 64)

    def get_global_model(self):
        return self.global_model

    def aggregate(self, client_params, weight):

        avg_params = {}

        for param_name in client_params[0].keys():
            avg_params[param_name] = torch.stack([weight * params[param_name] for params, weight in zip(client_params, weight)]).sum(dim=0)

        self.global_model.load_state_dict(avg_params)

    def get_global_params(self):
        return self.global_model.state_dict()

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