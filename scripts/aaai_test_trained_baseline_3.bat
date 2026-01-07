::SET BASED ON THE DATA LOCATIONS ON YOUR MACHINE

::PATH TO THE SRC FILES
set PYTHONPATH=..\src\

set orthomosaics_stats_file="H:\CRASAR-U-DROIDs\statistics.csv"
set orthomosaic_folder="H:\CRASAR-U-DROIDs\test\imagery\UAS"
set annotations_folder="..\data\test\annotations\road_damage_assessment"
set adjustments_folder="..\data\test\annotations\road_alignment_adjustments"
::SET BASED ON THE DATA LOCATIONS ON YOUR MACHINE

::PATH TO THE SRC FILES
set PYTHONPATH=..\src\

set orthomosaics_stats_file="H:\CRASAR-U-DROIDs\statistics.csv"
set orthomosaic_folder="H:\CRASAR-U-DROIDs\test\imagery\UAS"
set annotations_folder="..\data\test\annotations\road_damage_assessment"
set adjustments_folder="..\data\test\annotations\road_alignment_adjustments"

::Run Inference and Evaluation
set model_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\simple\Attention_UNet_Simple\epoch=12-step=6500.ckpt"
set hyperparameters_file_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\simple\Attention_UNet_Simple\RDA_UNet_simple_v1_unadjusted.yaml"
set preds_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\simple\Attention_UNet_Simple\AAAI26_unadj\preds_unadjusted.json"
set metrics_path="E:\Users\Thomas\Desktop\AAAI26-RDA-Models-Metrics-Preds\simple\Attention_UNet_Simple\AAAI26_unadj\metrics_unadjusted.json"

python ../src/modeling/infer.py --accelerator gpu ^
								--test_imagery_folder %orthomosaic_folder% ^
								--test_spatial_folder %annotations_folder% ^
								--preds_path %preds_path% ^
								--model_path %model_path% ^
								--test_adjustments_folder %adjustments_folder% ^
								--hyperparameters_yaml_path %hyperparameters_file_path% ^
								--ortho_stats_file %orthomosaics_stats_file%

python ../src/modeling/evaluate_RDA.py --road_lines_folder %annotations_folder% ^
									   --road_adjustments_folder %adjustments_folder% ^
									   --preds_path %preds_path% ^
									   --metrics_file %metrics_path% ^
									   --ortho_stats_file %orthomosaics_stats_file% ^
									   --hyperparameters_file %hyperparameters_file_path%