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

from utils.common_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,5,3"

version = "random_auxiliary_init_exp"
DEFAULT_CFG = {
    "random_layers": 1,
    "lambda_": 0.2,
    "auxiliary_start": 10,
    "progressive": True
}

baseline = {
    "random_layers": 0,
    "lambda_": 0,
    "auxiliary_start": 0,
    "progressive": False,
    "net_layers": 0,
    "with_auxiliary": False,
    "with_mixstyle": False
}


def ablation_exp(parameter_name, parameters, **kwargs):
    new_cfg = copy.copy(DEFAULT_CFG)
    for param in parameters:
        new_cfg[parameter_name] = param
        train_script(**kwargs, **new_cfg)


def train_script(random_layers=0, lambda_=0, superposition_start=149, auxiliary_type="Last", progressive=False,
                 dataset="SIM_10K_Dataset", weights="yolov5n.pt", batch_size=44, epochs=150,
                 net_layers=-1, superposition="MixStyle", limit=-1, cache=True, visualize=False, device=None,
                 address="127.0.1.1", auxiliary_initial_way='per-layer', init_before_batches=50,
                 initial_distribution_num=1, init_distribution_fixed=False,master_port=48200):
    if device is None:
        device = [0, 1]
    device = list(map(str, device))
    dataset_name = get_filename(dataset)
    log_dir = f"log/{version}"
    create_not_exist(log_dir)
    if auxiliary_type != "None" and superposition != "None":
        proj_name = f"@init_way:{auxiliary_initial_way}@num:{initial_distribution_num}{'@fixed' if init_distribution_fixed else ''}"
    else:
        proj_name = f"Default_{dataset_name}"
    cmd = f"nohup python  -m torch.distributed.launch  --nproc_per_node {len(device)} --master_addr {address}  --master_port {master_port} train.py" \
          f" --auxiliary-type {auxiliary_type} --superposition {superposition} --net-layers {net_layers}  {'--progressive' if progressive else ''}" \
          f" --data /data/yaser/data/research/{dataset}/data.yaml  --random-layers {random_layers} {'--cache' if cache else ''} " \
          f"{'--visualize' if visualize else ''}  --lambda_ {lambda_} --superposition-start {superposition_start} " \
          f"--initial-distribution-num {initial_distribution_num} --init-before-batches {init_before_batches} " \
          f"--random-auxiliary-initial-way {auxiliary_initial_way} {'--init-distribution-fixed' if init_distribution_fixed else ''}" \
          f" --hyp  hyp.VOC.yaml  --weights {weights} --batch-size {batch_size} " \
          f"  --label-smoothing 0.15 --img-size 960 --limit-data {limit} " \
          f" --device {','.join(device)} --project runs/{version}/{dataset_name} " \
          f"--name {proj_name} --epochs {epochs} >> {log_dir}/{proj_name}.log"

    debug(cmd)
    os.system(cmd)


if __name__ == '__main__':
    # for dataset in ["SIM_10K_Dataset", "CityscapeDataset_car/foggyDataset", "CityscapeDataset_car/rainDataset",
    #                 "CityscapeDataset_car/rawDataset"]:
    #     train_script(dataset=dataset, **baseline)  # 基准模型

    base_config = {
        "random_layers": 1,
        "lambda_": 0.1,
        "superposition_start": 5,
        "progressive": True,
        "net_layers": 2,
        "device": [4, 5],
        "auxiliary_type": "Random",
        "superposition": "MixStyle",
        "master_port": 48500,
        "address":"127.0.1.2"
    }

    for auxiliary_initial_way in ['per-layer', 'whole-net']:
            for initial_distribution_num in [3, 5, 7]:
                train_script(auxiliary_initial_way=auxiliary_initial_way,
                             initial_distribution_num=initial_distribution_num,
                             init_distribution_fixed=True, **base_config)
