import os
import glob
import argparse
import torch
import yaml
import tables as tb

from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from modeling.utils.data_augmentations import (
    get_valid_transforms,
    get_normalize_transform,
    get_tensor_transform,
    get_train_transforms,
)

from modeling.adaptors.WindowedDatasetAdaptor import WindowedDatasetAdaptor
from modeling.datasets.WindowedDataset import WindowedDataset
from modeling.data_modules.TrainValPredictDataModule import TrainValPredictDataModule
from modeling.DataMap import Labels2IdxMap
from modeling.Orthomosaic import OrthomosaicFactory
from modeling.Models.model_registry import (
    STR2MODELCLASS,
    STR2TASKMODELCLASS,
    LOCATIONSTRATEGY2MODULEMAPPING,
    MASKINGSTRATEGY2MODULEMAPPING,
    KEYPOINTSTRATEGY2MODULEMAPPING,
    SAMPLEANNOTATORTRATEGY2MODULEMPAPPING,
    PRESENTATIONTRATEGY2MODULEMAPPING
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
print(f"Running process {os.getpid()} on local rank {local_rank} and __name__ = {__name__}")

tb.parameters.MAX_BLOSC_THREADS = 1
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the specified model on the predefined task."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="The path to the folder where the metrics, logs, and checkpoints will be saved.",
        default="./",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="The path to the model checkpoint from which training should resume.",
        default=None,
    )
    parser.add_argument(
        "--data_gen_workers",
        type=int,
        help="The number of worker processes that will be used for data generation",
        default=12,
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="The floating point precision with which the model should be trained.",
        default="16-mixed",
    )
    parser.add_argument(
        "--accelerator",
        help="Which hardware accelerator should be used for model training (cpu, gpu, tpu, mps)",
        default="cpu",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When set, use simplified parameters to initialize things faster.",
    )
    parser.add_argument(
        "--train_ortho_path",
        type=str,
        help="The path to the folder containing orthomosaics that will be used for training",
    )
    parser.add_argument(
        "--train_hdf5_path",
        type=str,
        help="The path to the folder containing hdf5 files that will be used for training",
    )
    parser.add_argument(
        "--train_labels_path",
        type=str,
        help="The path to the folder containing labels that will be used for training",
    )
    parser.add_argument(
        "--train_adjustments_path",
        type=str,
        help="The path to the folder containing adjustments that will be used for training",
    )
    parser.add_argument(
        "--val_ortho_path",
        type=str,
        help="The path to the folder containing orthomosaics that will be used for validation",
    )
    parser.add_argument(
        "--val_hdf5_path",
        type=str,
        help="The path to the folder containing hdf5 files that will be used for validation",
    )
    parser.add_argument(
        "--val_labels_path",
        type=str,
        help="The path to the folder containing labels that will be used for validation",
    )
    parser.add_argument(
        "--val_adjustments_path",
        type=str,
        help="The path to the folder containing adjustments that will be used for validation",
    )
    parser.add_argument(
        "--hyperparameters_yaml_path",
        type=str,
        help="Path to the hyperparameters yaml file path.",
    )
    parser.add_argument(
        "--ortho_stats_file",
        type=str,
        help="The path to the statistics.csv file included with the dataset.",
    )
    parser.add_argument(
        "--train_backend",
        type=str,
        help="The backend used to read imagery during training",
        default="auto",
    )
    parser.add_argument(
        "--val_backend",
        type=str,
        help="The backend used to read imagery during validation",
        default="auto",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="The number of GPUs to use for training",
        default=1,
    )
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    # Load the hyperparameters file
    print("Reading hyperparameters...")
    hyperparameters = {}
    print(args)
    with open(args.hyperparameters_yaml_path, encoding="utf-8") as stream:
        try:
            hyperparameters = yaml.safe_load(stream)
            print(hyperparameters)
        except yaml.YAMLError as exc:
            print("Encountered error parsing hyperparameters...")
            print(exc)

    input_dataset_label_map = Labels2IdxMap(
        hyperparameters["channel_maps"]["input_dataset_class_2_idx_map"],
        hyperparameters["channel_maps"]["background_class_idx"],
    )

    val_bda_labels_path = None
    val_bda_adjustments_path = None
    train_bda_labels_path = None
    train_bda_adjustments_path = None
    if hyperparameters["task"] == "BDA" or hyperparameters["task"] == "BDAADJ":
        val_bda_labels_path = args.val_labels_path
        val_bda_adjustments_path = args.val_adjustments_path
        train_bda_labels_path = args.train_labels_path
        train_bda_adjustments_path = args.train_adjustments_path

    val_rda_labels_path = None
    val_rda_adjustments_path = None
    train_rda_labels_path = None
    train_rda_adjustments_path = None
    if hyperparameters["task"] == "RDA" or hyperparameters["task"] == "RDAADJ":
        val_rda_labels_path = args.val_labels_path
        val_rda_adjustments_path = args.val_adjustments_path
        train_rda_labels_path = args.train_labels_path
        train_rda_adjustments_path = args.train_adjustments_path


    # Get the orthomsaics that will be used for validation
    print("Loading Validation Orthomosaics...")
    validation_orthomosaics = OrthomosaicFactory(
        orthomosaic_folder=args.val_ortho_path,
        table_folder=args.val_hdf5_path,
        bda_adj_annotation_folder=val_bda_adjustments_path,
        bda_annotation_folder=val_bda_labels_path,
        rda_annotation_folder=val_rda_labels_path,
        rda_adj_annotation_folder=val_rda_adjustments_path,
        limit=(1 if args.debug else None),
        backend=args.val_backend,
        statistics_file_path=args.ortho_stats_file,
    )
    print("\tDone")

    print("Loading Train Orthomosaics...")
    train_orthomosaics = OrthomosaicFactory(
        orthomosaic_folder=args.train_ortho_path,
        table_folder=args.train_hdf5_path,
        bda_adj_annotation_folder=train_bda_adjustments_path,
        bda_annotation_folder=train_bda_labels_path,
        rda_annotation_folder=train_rda_labels_path,
        rda_adj_annotation_folder=train_rda_adjustments_path,
        limit=(2 if args.debug else None),
        backend=args.train_backend,
        statistics_file_path=args.ortho_stats_file,
    )
    print("\tDone")

    print("Initializing Gloabl Strategies...")
    print("\tInitializing Masking Strategy")
    masking_strategy_args = {}
    if "masking_strategy_parameters" in hyperparameters.keys():
        masking_strategy_args = hyperparameters["masking_strategy_parameters"]
    masking_strat = MASKINGSTRATEGY2MODULEMAPPING[hyperparameters["task"]](**masking_strategy_args)

    print("\tInitializing Keypoint Strategy")
    keypoint_strat = KEYPOINTSTRATEGY2MODULEMAPPING[hyperparameters["task"]]()

    # Initialize the training dataset based on task
    print("Initializing Validation Dataset...")
    print("\tInitializing Validation Sample Location Generation Strategy")
    validation_presentation_strategy_args = {}
    if "validation_presentation_strategy_parameters" in hyperparameters.keys():
        validation_presentation_strategy_args.update(hyperparameters["validation_presentation_strategy_parameters"])

    val_loc_pres_strat = PRESENTATIONTRATEGY2MODULEMAPPING[hyperparameters["validation_presentation_strategy"]](**validation_presentation_strategy_args)
    val_annotator = SAMPLEANNOTATORTRATEGY2MODULEMPAPPING[hyperparameters["task"]](**hyperparameters["validation_annotator_parameters"])
    validation_sample_location_strategy_args = {
        "annotator":val_annotator,
        "sample_location_presentation_strategy":val_loc_pres_strat,
        "orthomosaics":validation_orthomosaics
    }
    if "validation_location_parameters" in hyperparameters.keys():
        validation_sample_location_strategy_args.update(hyperparameters["validation_location_parameters"])
    validation_location_strat = LOCATIONSTRATEGY2MODULEMAPPING[hyperparameters["validation_location_strategy"]](**validation_sample_location_strategy_args)

    print("\tInitializing Validation Dataset Adaptor. This may take a moment as it can involve generating samples to send to the model...")
    validation_dataset_adaptor_args = {
        "orthomosaics": validation_orthomosaics,
        "label_map": input_dataset_label_map,
        "sample_location_generation_strategy": validation_location_strat,
        "keypoint_conversion_strategy": keypoint_strat,
        "mask_generation_strategy": masking_strat,
    }
    validation_dataset_adaptor_args.update(hyperparameters["validation_dataset_adaptor_parameters"])
    validation_adaptor = WindowedDatasetAdaptor(**validation_dataset_adaptor_args)

    print("\tInitializing Validation Dataset")
    validation_dataset = WindowedDataset(validation_adaptor,
                                         get_valid_transforms(),
                                         get_normalize_transform(),
                                         get_tensor_transform(),
                                         hyperparameters["input"]["normalized_inputs"])

    print("\tDone")

    # Initialize the training dataset based on task
    print("Initializing Train Dataset...")
    print("\tInitializing Train Sample Location Generation Strategy")
    train_presentation_strategy_args = {}
    if "train_presentation_strategy_parameters" in hyperparameters.keys():
        train_presentation_strategy_args.update(hyperparameters["train_presentation_strategy_parameters"])
    train_loc_pres_strat = SAMPLEANNOTATORTRATEGY2MODULEMPAPPING[hyperparameters["task"]](**hyperparameters["train_annotator_parameters"])
    train_annotator = PRESENTATIONTRATEGY2MODULEMAPPING[hyperparameters["train_presentation_strategy"]](**train_presentation_strategy_args)
    train_sample_location_strategy_args = {
        "annotator":train_loc_pres_strat,
        "sample_location_presentation_strategy":train_annotator,
        "orthomosaics":train_orthomosaics
    }
    if "train_location_parameters" in hyperparameters.keys():
        train_sample_location_strategy_args.update(hyperparameters["train_location_parameters"])
    train_location_strat = LOCATIONSTRATEGY2MODULEMAPPING[hyperparameters["train_location_strategy"]](**train_sample_location_strategy_args)

    print("\tInitializing Train Dataset Adaptor. This may take a moment as it can involve generating samples to send to the model...")
    train_dataset_adaptor_args = {
        "orthomosaics": train_orthomosaics,
        "label_map": input_dataset_label_map,
        "sample_location_generation_strategy": train_location_strat,
        "keypoint_conversion_strategy": keypoint_strat,
        "mask_generation_strategy": masking_strat,
    }
    train_dataset_adaptor_args.update(hyperparameters["train_dataset_adaptor_parameters"])
    train_adaptor = WindowedDatasetAdaptor(**train_dataset_adaptor_args)

    print("\tInitializing Train Dataset")
    train_dataset = WindowedDataset(train_adaptor,
                                    get_train_transforms(),
                                    get_normalize_transform(),
                                    get_tensor_transform(),
                                    hyperparameters["input"]["normalized_inputs"])
    print("\tDone")

    print("Initializing Train & Validation Data Module")
    dm = TrainValPredictDataModule(train_dataset = train_dataset,
                                   valid_dataset = validation_dataset,
                                   num_workers = 1 if args.debug else args.data_gen_workers,
                                   train_batch_size = hyperparameters["input"]["training_parameters"]["batch_size"],
                                   valid_batch_size = hyperparameters["input"]["validation_parameters"]["batch_size"])

    # Initialize the model
    print("Initializing model...")
    print("\tTask:", hyperparameters["task"])
    print("\tModel:", hyperparameters["model"])
    model = (STR2TASKMODELCLASS[hyperparameters["task"]])(
        hyperparameters=hyperparameters, val_orthomosaics=validation_orthomosaics
    )
    model.initialize_model(
        STR2MODELCLASS[hyperparameters["model"]](
            hyperparameters, model.input_channel_map, model.output_label_map
        ).get_model()
    )
    print("\tDone")

    # Initialize the model and and set it to checkpoint on the best observed valid_loss
    print("Training...")
    logger = TensorBoardLogger(
        os.path.join(args.out_path, "tb_logs"),
        name=str(model.getName()) + "_" + str(hyperparameters["task"]),
    )
    checkpoint_callback = model.configure_checkpoint()

    # To avoid exploding gradients
    gradient_clip_val = None
    if "gradient_clip_val" in hyperparameters["input"]["training_parameters"].keys():
        gradient_clip_val = hyperparameters["input"]["training_parameters"][
            "gradient_clip_val"
        ]

    trainer_args = {
        "gradient_clip_val":gradient_clip_val,
        "max_epochs":hyperparameters["input"]["training_parameters"]["max_epochs"],
        "callbacks":[checkpoint_callback],
        "default_root_dir":args.out_path,
        "precision":args.precision,
        "logger":logger,
        "accelerator":args.accelerator,
        "profiler":None,
        "accumulate_grad_batches":hyperparameters["input"]["training_parameters"]["grad_accumulation"],
        "num_sanity_val_steps":-1 if args.restart else 0
    }

    if args.num_gpus > 1:
        print("Enabling MultiGPU Training...")
        multi_gpu_args = {
            "strategy":"ddp",
            "devices":args.num_gpus,
        }
        trainer_args = trainer_args | multi_gpu_args

    trainer = Trainer(**trainer_args)

    ckpt_path = None
    # To handle resume training when preempted...
    if args.restart:
        print("Restarting Training...")
    else:
        # Find latest version for the checkpoint
        version_folders = sorted(
            glob.glob(
                os.path.join(
                    os.path.join(args.out_path, "tb_logs"),
                    str(model.getName()) + "_" + str(hyperparameters["task"]),
                    "version_*",
                )
            ),
            key=os.path.getctime,
        )
        if version_folders:
            latest_version = version_folders[-1]
            checkpoint_dir = os.path.join(latest_version, "checkpoints")
            checkpoint_files = sorted(
                glob.glob(os.path.join(checkpoint_dir, "*.ckpt")), key=os.path.getctime
            )
            if checkpoint_files:
                ckpt_path = checkpoint_files[-1]
                print("Found checkpoint, Resuming Training...")
                print("\tCheckpoint found at", ckpt_path)
        if ckpt_path is None:
            print("Did Not Find Checkpoint, Restarting Training...")

    # Fit the model.
    trainer.fit(model, dm, ckpt_path=ckpt_path)
