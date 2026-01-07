::SET BASED ON THE DATA LOCATIONS ON YOUR MACHINE

::PATH TO THE SRC FILES
set PYTHONPATH=..\src\

set orthomosaics_stats_file="H:\CRASAR-U-DROIDs\statistics.csv"
set orthomosaic_folder="H:\CRASAR-U-DROIDs\test\imagery\UAS"
set annotations_folder="..\data\test\annotations\road_damage_assessment"
set adjustments_folder="..\data\test\annotations\road_alignment_adjustments"

::Run Inference and Evaluation
set model_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\full\RDA_UNet_full_attention_CCE_full\epoch=20-step=10500.ckpt"
set hyperparameters_file_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\full\RDA_UNet_full_attention_CCE_full\RDA_UNet_full_unadjusted.yaml"
set preds_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\full\RDA_UNet_full_attention_CCE_full\AAAI26_unadj\epoch=20-step=10500.ckpt_ortho_LitMaskedUNetModelRDA_unadjusted_preds.json"
set metrics_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\full\RDA_UNet_full_attention_CCE_full\AAAI26_unadj\CRASAR-U-DROIDs_RDA_metrics_unadjusted.json"

python ../src/modeling/evaluate_RDA.py --road_lines_folder %annotations_folder% ^
									   --road_adjustments_folder %adjustments_folder% ^
									   --preds_path %preds_path% ^
									   --metrics_file %metrics_path% ^
									   --ortho_stats_file %orthomosaics_stats_file% ^
									   --hyperparameters_file %hyperparameters_file_path%

set model_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\full\RDA_UNet_full_attention_CCE_full\epoch=20-step=10500.ckpt"
set hyperparameters_file_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\full\RDA_UNet_full_attention_CCE_full\RDA_UNet_full_adjusted.yaml"
set preds_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\full\RDA_UNet_full_attention_CCE_full\AAAI26_adj\epoch=20-step=10500.ckpt_ortho_LitMaskedUNetModelRDA_preds.json"
set metrics_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\full\RDA_UNet_full_attention_CCE_full\AAAI26_adj\CRASAR-U-DROIDs_RDA_metrics_adjusted.json"


python ../src/modeling/evaluate_RDA.py --road_lines_folder %annotations_folder% ^
									   --road_adjustments_folder %adjustments_folder% ^
									   --preds_path %preds_path% ^
									   --metrics_file %metrics_path% ^
									   --ortho_stats_file %orthomosaics_stats_file% ^
									   --hyperparameters_file %hyperparameters_file_path%
