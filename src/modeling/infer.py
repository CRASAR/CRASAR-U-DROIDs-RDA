import os
import json
import argparse
import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

from modeling.Orthomosaic import OrthomosaicFactory
from modeling.adaptors.WindowedDatasetAdaptor import WindowedDatasetAdaptor
from modeling.datasets.WindowedDataset import WindowedDataset
from modeling.data_modules.TrainValPredictDataModule import TrainValPredictDataModule
from modeling.DataMap import Labels2IdxMap

from modeling.utils.data_augmentations import (
    get_inference_transforms,
    get_normalize_transform,
    get_tensor_transform,
)

from modeling.Models.model_registry import (
    STR2MODELCLASS,
    STR2TASKMODELCLASS,
    LOCATIONSTRATEGY2MODULEMAPPING,
    MASKINGSTRATEGY2MODULEMAPPING,
    KEYPOINTSTRATEGY2MODULEMAPPING,
    SAMPLEANNOTATORTRATEGY2MODULEMPAPPING,
    PRESENTATIONTRATEGY2MODULEMAPPING
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with Model.")
    parser.add_argument("--test_imagery_folder", type=str, help="Path to orthomosaics.")
    parser.add_argument("--test_spatial_folder", type=str, help="Path to polygons.")
    parser.add_argument("--test_adjustments_folder", type=str, help="Path to adjustments.")
    parser.add_argument("--test_boundaries_folder", type=str, help="Path to boundaries.")
    parser.add_argument("--hyperparameters_yaml_path", type=str, help="Path to the hyperparameter yaml file path.")
    parser.add_argument("--model_path", type=str, help="The path to trained model.")
    parser.add_argument("--preds_path", type=str, help="The path to file where predicitons will be stored.")
    parser.add_argument("--data_gen_workers", type=int, help="The number of worker processes that will be used for data generation", default=12)
    parser.add_argument("--precision", type=str, help="The floating point precision with which the model should use.", default="16-mixed")
    parser.add_argument("--accelerator", help="Which hardware accelerator should be used for model training (cpu, gpu, tpu, mps)", default="cpu")
    parser.add_argument("--batch_size", type=int, help="The size of the batch used during inference", default=2)
    parser.add_argument("--matmul_precision", type=str,
                        help='The precision to be used by CUDA tensor cores when available. Options are ("medium", "high")', default=None)
    parser.add_argument("--ortho_stats_file", type=str, help="The path to the statistics.csv file included with the dataset.")
    parser.add_argument("--default_epsg_int", type=int,
                        help="The integer used to identify the CRS of orthosmosaics loaded when their transforms are stored in a TFW file.", default=None)
    parser.add_argument("--scale_factor", type=float, help="scale factor to downsample/upsample the imagery", default=1.0)
    args = parser.parse_args()

    if args.matmul_precision:
        print("Setting matmul precision to " + args.matmul_precision)
        torch.set_float32_matmul_precision(args.matmul_precision)

    print("Inferencing Model stored at " + str(args.model_path))

    # Create the location where the predictions are going to be stored
    preds_folder, _ = os.path.split(args.preds_path)
    if not os.path.exists(preds_folder):
        os.makedirs(preds_folder, exist_ok=True)
        print("Created the directory to store the output: " + str(preds_folder))

    # Initialize the model
    print("Reading hyperparams...")
    hyperparams = {}
    with open(args.hyperparameters_yaml_path) as stream:
        try:
            hyperparams = yaml.safe_load(stream)
            print(hyperparams)
        except yaml.YAMLError as exc:
            print("Encountered error parsing inference hyperparams...")
            print(exc)

    print("Creating index maps")
    input_dataset_label_map = Labels2IdxMap(
        hyperparams["channel_maps"]["input_dataset_class_2_idx_map"],
        hyperparams["channel_maps"]["background_class_idx"],
    )

    print("Initializing model...")
    model = (STR2TASKMODELCLASS[hyperparams["task"]])(hyperparameters=hyperparams, device=args.accelerator)
    model.initialize_model(
        STR2MODELCLASS[hyperparams["model"]](
            hyperparams, model.input_channel_map, model.output_label_map
        ).get_model()
    )

    if args.model_path:
        print("Loading model from:", args.model_path)
        model.load(args.model_path)
    print("Model Loaded...\n\n")
    print(ModelSummary(model, max_depth=1))
    print("\n\n\n")

    test_bda_labels_path = None
    test_bda_adjustments_path = None
    if hyperparams["task"] == "BDA" or hyperparams["task"] == "BDAADJ":
        test_bda_labels_path = args.test_spatial_folder
        test_bda_adjustments_path = args.test_adjustments_folder

    test_rda_labels_path = None
    test_rda_adjustments_path = None
    if hyperparams["task"] == "RDA":
        test_rda_labels_path = args.test_spatial_folder
        test_rda_adjustments_path = args.test_adjustments_folder

    print("Initializing Prediction Data Module")
    inference_orthomosaics = OrthomosaicFactory(
        orthomosaic_folder=args.test_imagery_folder,
        table_folder=None,
        boundary_folder=args.test_boundaries_folder,
        bda_annotation_folder=test_bda_labels_path,
        bda_adj_annotation_folder=test_bda_adjustments_path,
        rda_adj_annotation_folder=test_rda_adjustments_path,
        rda_annotation_folder=test_rda_labels_path,
        backend="rasterio",
        statistics_file_path=args.ortho_stats_file,
        default_epsg_int=args.default_epsg_int,
        scale_factor=args.scale_factor
    )

    print("Initializing Strategies...")
    print("\tInitializing Masking Strategy")
    masking_strategy_args = {}
    if "masking_strategy_parameters" in hyperparams.keys():
        masking_strategy_args = hyperparams["masking_strategy_parameters"]
    masking_strat = MASKINGSTRATEGY2MODULEMAPPING[hyperparams["task"]](**masking_strategy_args)

    print("\tInitializing Keypoint Strategy")
    keypoint_strat = KEYPOINTSTRATEGY2MODULEMAPPING[hyperparams["task"]]()

    print("\tInitializing Validation Sample Location Generation Strategy")
    validation_presentation_strategy_args = {}
    if "validation_presentation_strategy_parameters" in hyperparams.keys():
        validation_presentation_strategy_args.update(hyperparams["validation_presentation_strategy_parameters"])
    loc_pres_strat = PRESENTATIONTRATEGY2MODULEMAPPING[hyperparams["validation_presentation_strategy"]](**validation_presentation_strategy_args)
    sample_location_strategy_args = {
        "annotator":SAMPLEANNOTATORTRATEGY2MODULEMPAPPING[hyperparams["task"]](**hyperparams["validation_annotator_parameters"]),
        "sample_location_presentation_strategy":loc_pres_strat,
        "orthomosaics":inference_orthomosaics
    }   
    if "validation_location_parameters" in hyperparams.keys():
        sample_location_strategy_args.update(hyperparams["validation_location_parameters"])
    location_strat = LOCATIONSTRATEGY2MODULEMAPPING[hyperparams["validation_location_strategy"]](**sample_location_strategy_args)

    print("\tInitializing Dataset Adaptor. This may take a moment as it can involve generating samples to send to the model...")
    dataset_adaptor_args = {
        "orthomosaics": inference_orthomosaics,
        "label_map": input_dataset_label_map,
        "sample_location_generation_strategy": location_strat,
        "keypoint_conversion_strategy": keypoint_strat,
        "mask_generation_strategy": masking_strat,
    }
    dataset_adaptor_args.update(hyperparams["validation_dataset_adaptor_parameters"])
    adaptor = WindowedDatasetAdaptor(**dataset_adaptor_args)

    print("\tInitializing Dataset")
    dataset = WindowedDataset(adaptor, get_inference_transforms(), get_normalize_transform(), get_tensor_transform(), hyperparams["input"]["normalized_inputs"])
    print("\tDone")

    print("Initializing Prediction Data Module")
    dm = TrainValPredictDataModule(predict_dataset=dataset,
                                   num_workers=args.data_gen_workers,
                                   predict_batch_size=hyperparams["input"]["validation_parameters"]["batch_size"])

    trainer = Trainer(precision=args.precision, accelerator=args.accelerator, num_sanity_val_steps=0)

    # Predict the model.
    print("Starting Inference")
    trainer.predict(model, dm)
    preds = model.get_predicted_labels()

    # Write the predictions to a file
    print("Writing predictions...")
    with open(args.preds_path, "w") as f:
        f.write(json.dumps({"model_name": model.getName(), "preds": preds}))
    f.close()

    print("Done")
