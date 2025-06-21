
data_name=Roboflow_2
trainings=CD_base_resnet18_Roboflow_2_b1_lr0.01_train_val_100_linear_1_1
pred_masks=pred
gt_masks=./datasets/${data_name}/label
original_imgs=./datasets/${data_name}/B
blended_mask=testing_analysis

python eval_instances.py --training_name ${trainings} --pred_masks ${pred_masks} --gt_masks ${gt_masks} --original_imgs ${original_imgs} --blended_mask ${blended_mask}