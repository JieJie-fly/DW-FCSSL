# -*-coding:utf-8-*-
from utils.Client import Client
from utils.server import Server
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def federated_learning(num_clients, rounds, num_epochs, dataset):
    # 定义源域模型
    clients = [Client(dataset[i]) for i in range(num_clients)]
    # 定义Server模型
    server = Server()

    for round in range(rounds):
        client_params = []
        global_model = server.get_global_model()

        training_Loss = []
        print(f"第 {round + 1} 轮全局通信")
        for index, client in enumerate(clients):
            # begin1 = time.time()
            client_training_loss = client.train(num_epochs, global_model)  # 训练每个源模型
            # end1 = time.time()
            # time_cost1 = end1 - begin1
            # print('Local extractor training time:', time_cost1)

            client_params.append(client.get_params())
            training_Loss.append(client_training_loss)

        total = np.sum(1 / i for i in training_Loss)
        weights = [(1 / i) / total for i in training_Loss]
        server.aggregate(client_params, weights)
    print('training complete')

    return server

