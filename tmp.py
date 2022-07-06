#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：yolov5-master 
@File    ：tmp.py
@IDE     ：PyCharm 
@Author  ：Yaser
@Date    ：4/10/2022 10:07 AM 
@Describe: 
"""
import copy
import math

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def action(x):
    return math.log2(3 * math.log(x ** 3) + 5 * x ** 2 + 2 * x + 15)


x = np.array(range(1, 500))
y = np.array([action(i) for i in x])
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


net = Net()
net2 = Net()
net2.requires_grad_(False)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
net = net.cuda(0)
net.requires_grad_(True)
net.train()
loss_func = nn.MSELoss()
epochs = 200
# tmp_state_dict = copy.deepcopy(net.state_dict())
for epoch in tqdm(range(epochs)):
    _input = torch.from_numpy(x).float().cuda(0)
    _input = _input.view(-1, 1)
    prediction = net(_input)
    prediction = prediction.cpu()
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch == 50:
        net2.load_state_dict(net.state_dict())

net.eval()
net2.eval()
net2.cuda(0)
x_test = np.array(range(500, 550))
x_test = torch.tensor(x_test, dtype=torch.float32).view(-1, 1).cuda(0)
prediction = net(x_test)
prediction = prediction.cpu().detach().numpy()
prediction2 = net2(x_test)
prediction2 = prediction2.cpu().detach().numpy()
x_test = x_test.cpu().detach().numpy()
y_test = np.array([action(i) for i in x_test])

for i in range(len(x_test)):
    print(f"{x_test[i]}: {prediction[i]} {prediction2[i]} {y_test[i]}")
