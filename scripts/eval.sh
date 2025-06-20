#!/usr/bin/env bash

gpus=0

data_name=Roboflow_2 #Biemme_4 #Biemme_3 #DSIFN
img_size=512  #1024
batch_size=1 #8
net_G=base_resnet18 #base_transformer_pos_s4_dd8
split=test
project_name=CD_base_resnet18_Roboflow_2_b1_lr0.01_train_val_100_linear #CD_base_transformer_pos_s4_dd8_DSIFN_b8_lr0.01_train_val_5_linear
checkpoint_name=best_ckpt.pt
output_folder=output_analysis/${project_name}/pred

python eval_cd.py --split ${split} --img_size ${img_size} --batch_size ${batch_size} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --output_folder ${output_folder}
 

