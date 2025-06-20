#output_analysis/CD_base_transformer_pos_s4_dd8_dedim8_Roboflow_1_b1_lr0.01_train_val_20_linear_2
#output_analysis/CD_base_transformer_pos_s4_dd8_dedim8_Biemme_3_b1_lr0.01_train_val_100_linear_2
#vis/CD_base_transformer_pos_s4_dd8_dedim8_Biemme_4_b1_lr0.01_train_val_20_linear_2

data_name=Roboflow_2
trainings=CD_base_resnet18_Roboflow_2_b1_lr0.01_train_val_100_linear
pred_masks=pred
gt_masks=./datasets/${data_name}/label
original_imgs=./datasets/${data_name}/B
blended_mask=testing_analysis

python eval_instances.py --training_name ${trainings} --pred_masks ${pred_masks} --gt_masks ${gt_masks} --original_imgs ${original_imgs} --blended_mask ${blended_mask}