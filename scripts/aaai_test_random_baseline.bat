::SET BASED ON THE DATA LOCATIONS ON YOUR MACHINE

::PATH TO THE SRC FILES
set PYTHONPATH=\src\


set orthomosaics_stats_file="<PATH_TO_CRASAR_U_DROIDS_IMAGERY>\statistics.csv"
set orthomosaic_folder="<PATH_TO_CRASAR_U_DROIDS_IMAGERY>\test\imagery\UAS"
set annotations_folder="\data\test\annotations\road_damage_assessment"
set adjustments_folder="\data\test\annotations\road_alignment_adjustments"

::Trained models result from running the "Train_Baseline_Slurm_Jobfile" in /scripts
set model_path="<PATH_TO_TRAINED_MODEL>"

::Select the hyperparameters file used to train the model
set hyperparameters_file_path="<PATH_TO_HYPERPARAMETERS_FILE>"

set preds_path="<PATH_TO_DESIRED_OUTPUT_FOLDER>\preds_adjusted.json"
set metrics_path_random="<PATH_TO_DESIRED_OUTPUT_FOLDER>\metrics_adjusted.json"
set plots_folder="<PATH_TO_DESIRED_OUTPUT_FOLDER>\plots"


::USE OF THE "--random_baseline" FLAG ENGAGES THE RANDOM BASELINE, CONVERTING ALL PREDICTIONS MADE BY THE PASSED MODEL TO RANDOM LABELS

python ../../../../src/modeling/evaluate_RDA.py --random_baseline ^
												--road_lines_folder %annotations_folder% ^
										        --road_adjustments_folder %adjustments_folder% ^
										        --preds_path %preds_path% ^
										        --metrics_file %metrics_path_random% ^
										        --ortho_stats_file %orthomosaics_stats_file% ^
										        --hyperparameters_file %hyperparameters_file_path%

python ../../../../src/modeling/formatters/plot_metrics_RDA.py --metrics_files %metrics_path_random% ^
													           --plots_folder %plots_folder%