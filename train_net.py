#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：yolov5-master 
@File    ：train_net.py
@IDE     ：PyCharm 
@Author  ：Yaser
@Date    ：3/14/2022 10:46 PM 
@Describe: 
"""
import copy

from utils.common_utils import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "3,0,1,2"

version = "V5_5_with_MixStyle_0615"
DEFAULT_CFG = {
    "random_layers": 1,
    "lambda_": 0.2,
    "auxiliary_start": 10,
    "progressive": True
}


def ablation_exp(parameter_name, parameters, **kwargs):
    new_cfg = copy.copy(DEFAULT_CFG)
    for param in parameters:
        new_cfg[parameter_name] = param
        train_script(**kwargs, **new_cfg)


def train_script(random_layers=0, lambda_=0, auxiliary_start=149, with_auxiliary=True, progressive=False,
                 dataset="SIM_10K_Dataset", weights="yolov5n.pt", batch_size=44, epochs=150,
                 net_layers=-1, with_mixstyle=True):
    dataset_name = get_filename(dataset)
    log_dir = f"log/{version}"
    create_not_exist(log_dir)
    if with_auxiliary and with_mixstyle:
        proj_name = f"@R_{random_layers}@Lay_{net_layers}@Lmd_{lambda_}@mixstyle"
    elif with_mixstyle:
        proj_name = f"Default_{dataset_name}_@R_{random_layers}@Lay_{net_layers}@Lmd_{lambda_}@mixstyle"
    else:
        proj_name = f"Default_{dataset_name}"
    cmd = f"nohup python  -m torch.distributed.launch  --nproc_per_node 2 --master_addr 127.0.1.2  --master_port 49200 train.py {'--with-mixstyle' if with_mixstyle else ''} --net-layers {net_layers}  {'--progressive' if progressive else ''} --data /data/yaser/data/research/{dataset}/data.yaml {'--with-auxiliary' if with_auxiliary else ''} --random-layers {random_layers} --cache  --lambda_ {lambda_} --auxiliary-start {auxiliary_start}  --hyp  hyp.VOC.yaml  --weights {weights} --batch-size {batch_size}   --label-smoothing 0.15 --img-size 960  --device 0,3 --project runs/{version}/{dataset_name} --name {proj_name} --epochs {epochs} >> {log_dir}/{proj_name}.log"

    # cmd = f"python  -m torch.distributed.launch  --nproc_per_node 2 --master_addr 127.0.1.2  --master_port 39200 train.py --limit-data 100 {'--progressive' if progressive else ''} --data /data/yaser/data/research/{dataset}/data.yaml {'--auxiliary-shuffle' if shuffle else ''} {'--with-auxiliary' if with_auxiliary else ''} --random-layers {random_layers if with_auxiliary else 0} --cache  --lambda_ {lambda_ if with_auxiliary else 0} --auxiliary-start {auxiliary_start if with_auxiliary else 999}  --hyp  hyp.VOC.yaml  --weights {weights} --batch-size {batch_size}   --label-smoothing 0.15 --img-size 960  --device 0,1 --project runs/{version}/{dataset_name} --name {proj_name} --epochs 150"
    debug(cmd)
    os.system(cmd)


if __name__ == '__main__':
    # train_script(with_auxiliary=False, weights="yolov5m.pt")  # 默认模型

    # train_script(progressive=True, lambda_=0.2, auxiliary_start=10, random_layers=1)
    # default_sim_weight = "/data/yaser/project/yolo_random/runs/Default/Default_SIM_10K_Dataset_car/weights/best.pt"
    # train_script(progressive=True, shuffle=True, lambda_=0.2, auxiliary_start=3, random_layers=1,
    #              weights=default_sim_weight, epochs=50)  # 默认模型
    # 测试层数的影响
    # train_script(progressive=True, lambda_=0.1, auxiliary_start=5, net_layers=2, with_auxiliary=False)  # 默认模型
    base_parameter = {
        "random_layers": 1,
        "lambda_": 0.1,
        "auxiliary_start": 5,
        "progressive": True,
        "net_layers": 2,
        "with_auxiliary": True,
        "with_mixstyle": True
    }
    # 测试层数影响
    for num in range(1, 6):
        for random_num in range(1, num + 1):
            if num == 2 and random_num == 1:
                continue
            parameter_copy = copy.deepcopy(base_parameter)
            parameter_copy["net_layers"] = num
            parameter_copy["random_layers"] = random_num
            train_script(**parameter_copy)  # 有辅助网络
            parameter_copy["with_auxiliary"] = False
            train_script(**parameter_copy)  # 无辅助网络

    for dataset in ["SIM_10K_Dataset", "CityscapeDataset_car/foggyDataset", "CityscapeDataset_car/rainDataset",
                    "CityscapeDataset_car/rawDataset"]:
        train_script(dataset=dataset, with_mixstyle=False, with_auxiliary=False)  # 基准模型

    # for _lambda in np.arange(0.05, 0.55, 0.05):
    #     if _lambda == 0.1:
    #         continue
    #     parameter_copy = copy.deepcopy(base_parameter)
    #     parameter_copy["lambda_"] = _lambda
    #     train_script(**parameter_copy)  # 有辅助网络
    #     parameter_copy["with_auxiliary"] = False
    #     train_script(**parameter_copy)  # 无辅助网络

    # for layer in range(2, 20):
    #     train_script(progressive=True, shuffle=True, lambda_=0.1, auxiliary_start=10, net_layers=layer)  # 默认模型
    # train_script(progressive=True, shuffle=True, lambda_=0.4, auxiliary_start=10, random_layers=2)  # 默认模型
    # train_script(progressive=True, shuffle=True, lambda_=0.2, auxiliary_start=3, random_layers=1)  # 默认模型
    # train_script(progressive=True, shuffle=True, lambda_=0.7, auxiliary_start=10, random_layers=1)  # 默认模型

    # for i in np.arange(0.05, 0.55, 0.05):
    #     train_script(progressive=True, shuffle=True, lambda_=round(i, 2), auxiliary_start=10, net_layers=2)  # 默认模型
    #
    # for i in range(2, 5):
    #     train_script(dataset="CityscapeDataset/rawDataset", progressive=True, shuffle=True, lambda_=0.2,
    #                  auxiliary_start=10, random_layers=i)  # 默认模型
    # for i in range(2, 5):
    #     train_script(dataset="CityscapeDataset/rawDataset", progressive=True, shuffle=True, lambda_=0.3,
    #                  auxiliary_start=10, random_layers=i)  # 默认模型
    # train_script(progressive=True, shuffle=True, lambda_=0.2, auxiliary_start=5, random_layers=1)
    # # 验证层数的影响
    # train_script(progressive=True, shuffle=True, lambda_=0.2, auxiliary_start=5, random_layers=2)
    # # 验证lambda的影响
    # train_script(progressive=True, shuffle=True, lambda_=0.4, auxiliary_start=5, random_layers=2)

    # train_script(dataset="CityscapeDataset/foggyDataset", progressive=True, lambda_=0.2, auxiliary_start=10)  # 默认模型
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # ablation_exp_cfg = {
    #     "random_layers": [i for i in range(1, 9)],
    #     "lambda_": [round(i, 2) for i in np.arange(0.1, 0.5, 0.05)],
    #     # "auxiliary_start": [i for i in range(5, 40, 5)]
    # }
    # for parameter_name, parameters in ablation_exp_cfg.items():
    #     info(f"==========ABLATION_EXP {parameter_name}==========")
    #
    #     ablation_exp(parameter_name, parameters, dataset="CityscapeDataset/rawDataset", weights="yolov5s.pt")
