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
import os.path

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

from utils.common_utils import *

DATASET_MAP = {
    "Cityscape_raw": "/data/yaser/data/research/CityscapeDataset/rawDataset/data.yaml",
    "Cityscape_rain": "/data/yaser/data/research/CityscapeDataset/rainDataset/data.yaml",
    "Cityscape_foggy": "/data/yaser/data/research/CityscapeDataset/foggyDataset/data.yaml"
}

DATASET_CAR_MAP = {
    "Cityscape_rain_car": "/data/yaser/data/research/CityscapeDataset_car/rainDataset/data.yaml",
    "Cityscape_raw_car": "/data/yaser/data/research/CityscapeDataset_car/rawDataset/data.yaml",
    "Cityscape_foggy_car": "/data/yaser/data/research/CityscapeDataset_car/foggyDataset/data.yaml",
    "SIM_10K_car": "/data/yaser/data/research/SIM_10K_Dataset/data.yaml",

}


def val_script(proj_name, weight_path, dataset_path, dataset_name, val_name):
    if exist(weight_path):
        log_path = f"val_log/{val_name}/{dataset_name}"

        create_not_exist(log_path)
        cmd = f"nohup python val.py --data {dataset_path} --weights {weight_path} --img 960 --device 0 --augment --verbose --project runs/val/{val_name}/{dataset_name} --name {proj_name} >> {log_path}/{proj_name}.log"
        debug(cmd)
        os.system(cmd)
    else:
        error(f"{weight_path} is not exist!")


def test_dir(dir_path, val_name, dataset_map=None):
    if dataset_map is None:
        dataset_map = DATASET_MAP
    exp_list = dir_list(dir_path)
    for exp_path in exp_list:
        proj_name = exp_path.split("/")[-1]
        weight_path = connect_path(exp_path, "weights", "best.pt")
        if exist(weight_path):
            for dataset_name, dataset_path in dataset_map.items():
                val_script(proj_name, weight_path, dataset_path, dataset_name, val_name)


def test_one(dir_path, val_name, dataset_map=None):
    if dataset_map is None:
        dataset_map = DATASET_MAP
    proj_name = dir_path.split("/")[-1]
    weight_path = connect_path(dir_path, "weights", "best.pt")
    if exist(weight_path):
        for dataset_name, dataset_path in dataset_map.items():
            val_script(proj_name, weight_path, dataset_path, dataset_name, val_name=val_name)


if __name__ == '__main__':
    exp_path = "/data/yaser/project/yolo_auxiliary/runs/2022_7_21/SIM_10K_Dataset/@R_2@Lay_2@Lmd_0.1@aux_Random@sup_MixStyle"
    test_one(exp_path, val_name="2022_7_22", dataset_map=DATASET_CAR_MAP)
    # test_dir(exp_path, val_name="test", dataset_map=DATASET_CAR_MAP)
