run_train(){
  python -m torch.distributed.launch --nproc_per_node 2 train.py --data /data/yaser/data/research/SIM_10K_Dataset/data.yaml  --hyp  hyp.finetune.yaml  --weights yolov5n6.pt --batch-size 20  --image-weights --cache --label-smoothing 0.15 --img-size 960  --device 0,1 --project runs/power/ --name hxq --epochs 150
}


