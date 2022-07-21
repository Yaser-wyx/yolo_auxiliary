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
import torch

from models.yolo import Model
from utils.common_utils import load_config


_lambda = torch.distributions.Beta(0.5, 0.71).sample((5, 1, 1, 1))
more_than = _lambda > 0.5
print(_lambda)
_lambda[more_than] = 1 - _lambda[more_than]
print(_lambda)
