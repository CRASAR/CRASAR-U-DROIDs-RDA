import time
import json
from collections import defaultdict
from torch import optim

import pytorch_lightning as L
import torch
import numpy as np

from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from modeling.utils.data_augmentations import (
    get_tensor_transform,
    get_normalize_transform,
)
from modeling.DataMap import Labels2IdxMap, ColorMap, Channel2IdxMap
from modeling.utils.inspection_utils import inspect_image, inspect_grad_flow
from modeling.utils.loss_utils import WeightedLoss, get_ipw_weights_from_class_counts, get_log_class_balanced_weights_from_class_counts
from modeling.ModelStepMetadata import ModelStepMetadata
from modeling.constants import POLYGON_COUNT_PREFIX, PIXEL_COUNT_PREFIX, SAMPLE_GENERATION_TIMING_PREFIX, SAMPLE_METADATA_ATTEMPTS, SAMPLE_METADATA_EXCEPTIONS
from modeling.Models.Baselines.Baseline import BaselineModel


class BaseModel(L.LightningModule):
    # pylint: disable-next=too-many-branches
    def __init__(self, hyperparameters=None, val_orthomosaics=None, device="cuda", quantiles=None):
        super().__init__()

        self._device = device
        self._qs = quantiles if not quantiles is None else [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        self.hyperparameters = hyperparameters

        self.dataset_label_map = Labels2IdxMap(
            hyperparameters["channel_maps"]["input_dataset_class_2_idx_map"],
            hyperparameters["channel_maps"]["background_class_idx"],
        )
        self.input_channel_map = Channel2IdxMap(hyperparameters["input"]["channels"])
        self.input_background_idx = self.input_channel_map.getIdx("mask")
        self.output_background_idx = hyperparameters["channel_maps"]["background_class_idx"]

        self.output_label_map = Labels2IdxMap(
            hyperparameters["channel_maps"]["output_class_2_idx_map"],
            hyperparameters["channel_maps"]["background_class_idx"],
        )
        self.idx2color_map = ColorMap(
            hyperparameters["channel_maps"]["model_class_2_color_map"],
            hyperparameters["channel_maps"]["output_class_2_idx_map"],
        )

        self.default_label = hyperparameters["channel_maps"]["default_label"]
        self._running_class_counts = {k:0 for k in self.output_label_map.getAllLabels()}

        try:
            self._normalized_inputs = hyperparameters["input"]["normalized_inputs"]
        except KeyError:
            self._normalized_inputs = False
        try:
            self.l1_reg = float(hyperparameters["input"]["training_parameters"]["l1_reg"])
        except KeyError:
            self.l1_reg = None
        try:
            self.l2_reg = float(hyperparameters["input"]["training_parameters"]["l2_reg"])
        except KeyError:
            self.l2_reg = None
        try:
            self.gamma = float(hyperparameters["input"]["training_parameters"]["gamma"])
        except KeyError:
            self.gamma = None
        try:
            self.alpha = float(hyperparameters["input"]["training_parameters"]["alpha"])
        except KeyError:
            self.alpha = None
        try:
            self.lr = float(hyperparameters["input"]["training_parameters"]["optimizer_parameters"]["learning_rate"])
        except KeyError:
            self.lr = None
        try:
            self._log_images_every_n_steps = hyperparameters["input"]["training_parameters"]["log_images_every_n_steps"]
        except KeyError:
            self._log_images_every_n_steps = None
        try:
            self.criterion = WeightedLoss(hyperparameters["input"]["training_parameters"]["loss_parameters"],
                                          None,
                                          ignore_index=self.output_background_idx)
            
        except KeyError:
            self.criterion = None
        try:
            self._include_gsd = hyperparameters["input"]["model_parameters"]["encoder_parameters"]["backbone"] == "scalemae"
        except KeyError:
            self._include_gsd = False
        try:
            self._include_mask_input = hyperparameters["input"]["mask_input"]
        except KeyError:
            self._include_mask_input = True # default to passing in mask as an input to the model
        try:
            self.criterion_scale_factor = hyperparameters["input"]["training_parameters"]["loss_parameters"]["scale_factor"]
        except KeyError:
            self.criterion_scale_factor = 1.0

        self._cur_iter = 0
        self._cur_step = 0
        self._prev_start_time = time.time()

        self._logger = None

        self._step_metadata = ModelStepMetadata(self.global_step)
        self._images_logged = False
        self._reset_aggregation_step_metadata()

        self.tensor_transform = get_tensor_transform()
        self.normalize_transform = get_normalize_transform()

        self.val_orthomosaics = val_orthomosaics

        self.validation_step_outputs = defaultdict(list)
        self.validation_step_labels = {}
        self.validation_loss = []
        self.predict_step_outputs = defaultdict(list)
        self.predicted_labels = {}

        self._name = hyperparameters["name"]
        self._hyperparameters = hyperparameters

        # Initialize Model
        self.model = None


    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()
        self.validation_loss.clear()

        for c in self.output_label_map.getAllLabels():
            self._step_metadata.scalars["val/Predicted_Pixel_Counts"][c] = 0
            
        self.model.eval()

    def get_l1_loss(self):
        # Get the L1 losses from the model
        l1_reg_term = 0
        for param in self.model.parameters():
            l1_reg_term += 0.5 * torch.sum(param**2)
        l1_reg_loss = self.l1_reg * l1_reg_term
        return l1_reg_loss
    

    def on_train_start(self):
        self.get_tb_logger().add_text("hyperparameters", json.dumps(self.hyperparameters), global_step=0)

    def log_batch_telemetry(self, batch):

        #Log all the data from fields where will have multiple scalars
        for key in batch.getMetadataKeys():
            scalars_collection = None
            if POLYGON_COUNT_PREFIX in key:
                scalars_collection = "Statistics/Polygon Statistics"
            elif PIXEL_COUNT_PREFIX in key:
                scalars_collection = "Statistics/Pixel Statistics"
            elif SAMPLE_GENERATION_TIMING_PREFIX in key:
                scalars_collection = "Timing/Sample Generating Times"
            if scalars_collection:
                self._step_metadata.scalars[scalars_collection][key] += sum(batch.getBatchedMetadataEntry(key))
        self._step_metadata.normalizations["Timing/Sample Generating Times"] += len(batch)
        self._step_metadata.normalizations["Statistics/Polygon Statistics"] += len(batch)
        self._step_metadata.normalizations["Statistics/Pixel Statistics"] += len(batch)

        #Then we can work through the fields with the single values
        self._step_metadata.scalar["Statistics/Sample Generator Attempts"] += sum(batch.getBatchedMetadataEntry(SAMPLE_METADATA_ATTEMPTS))
        self._step_metadata.normalizations["Statistics/Sample Generator Attempts"] += len(batch)

        #Finally, we can log the exceptions
        for entry in batch.getBatchedMetadataEntry(SAMPLE_METADATA_EXCEPTIONS):
            for exception_name, count in entry.items():
                self._step_metadata.scalars["Statistics/Monitored Exceptions"][exception_name] += count

    def getName(self):
        return self._name

    def initialize_model(self, model):
        self.model = model

    def forward(self, batch):  # pylint: disable=arguments-differ
        return self.model.forward(batch)

    def get_predicited_labels(self):
        return self.predicted_labels

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch.moveTo(device, move_raw_imagery=False)
        return batch

    def format_batched_sample_for_model(self, batch):
        rgb = batch.getBatchedImagery()
        mask = batch.getBatchedQueries().unsqueeze(1)

        hyp_channels = self._hyperparameters["input"]["channels"]
        actual_channels = ["red", "green", "blue", "mask"]

        try:
            if not self._include_mask_input:
                hyp_channels.pop("mask")
        except KeyError:
            pass

        channels_sorted = sorted(hyp_channels.items(), key=lambda x: x[1])
        permute = [actual_channels.index(c[0]) for c in channels_sorted]
        if any(p < 0 for p in permute) or any(p > 3 for p in permute):
            raise ValueError("Hyperparamters passed field that is not available in the batch.")

        if self._include_gsd:
            gsd_batched = []
            for gsd in batch.getBatchedGSD():
                gsd_batched.append(gsd[0] / 100)

            concatenated = torch.cat((rgb, mask), dim=1)
            if self._include_mask_input:
                return (
                    concatenated,
                    gsd_batched,
                )  # TODO: Consider how to handle this more gracefully...
            return (
                concatenated[:, permute],
                gsd_batched,
                concatenated[:, self.input_background_idx],
            )  # Mask separate for masking TODO: There is a better way to do this ...

        if not self._include_mask_input:
            return (
                torch.cat((rgb, mask), dim=1)[:, permute],
                torch.cat((rgb, mask), dim=1)[:, self.input_background_idx],
            )
        return torch.cat((rgb, mask), dim=1)[:, permute]

    def load(self, path, strict=True):
        self.model.load(path, strict)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.model.load_state_dict(state_dict, strict, assign)

    def _reset_aggregation_step_metadata(self):
        self._step_metadata = ModelStepMetadata(self.global_step)
        self.mark_images_logged(False)

    def _add_batched_images_to_step_metadata(self, batched_images, name, inspection_func=inspect_image, image_limit=-1):
        #Count up the number of occurances of this name so we dont overwrite data in the object
        cur_keys_in_meta = self._step_metadata.images.keys()
        cur_id = 0
        for field_key in cur_keys_in_meta:
            if f"{name}/" in field_key:
                cur_id += 1

        #With the correct ID now known, now we can add the image
        images = inspection_func(batched_images)
        for img_idx, image in enumerate(images):
            if image_limit == -1 or (img_idx + cur_id) < image_limit:
                field_idx = cur_id + img_idx
                self._step_metadata.images[f"{name}/{field_idx}"] = image

     # pylint: disable-next=too-many-branches
    def _log_labels_update_loss(self, batched_labels=None):
        if not batched_labels is None:
            unique_values, counts = torch.unique(batched_labels, return_counts=True)
        else:
            unique_values, counts = [], []

        class_counts = {}
        sum_counts = sum(counts)
        for unique_value, count in zip(unique_values, counts):
            labels = self.output_label_map.getLabels(int(unique_value))
            if len(labels) > 0:
                class_counts[labels[0]] = count/sum_counts
            else:
                print("Warning: Found an index without a Label", labels, unique_value, count)

        for key, count in class_counts.items():
            self._running_class_counts[key] += count

        cw = [1] * len(self.output_label_map)
        strategy_name = self.hyperparameters["input"]["training_parameters"]["output_class_weights_strategy"].lower()

        if strategy_name == "uniform":
            self.class_weights = torch.tensor(cw).to(self._device)

        elif strategy_name == "manual":
            if len(self.hyperparameters["input"]["training_parameters"]["output_class_weights"]) != len(self.output_label_map):
                print(
                    "Warning: Found a different number of class weights vs output indicies in model hyperparameter."
                    + "This may result in some output classes being weighted incorrectly during training."
                )
            for label, weight in self.hyperparameters["input"]["training_parameters"]["output_class_weights"].items():
                cw[self.output_label_map.getIndex(label)] = weight

            self.class_weights = torch.tensor(cw).to(self._device)

        elif strategy_name in ["ipw", "log_class_balance"]:
            for label, count in self._running_class_counts.items():
                cw[self.output_label_map.getIndex(label)] = count
            if strategy_name == "ipw":
                self.class_weights = torch.tensor(get_ipw_weights_from_class_counts(cw)).to(self._device)
            else:
                self.class_weights = torch.tensor(get_log_class_balanced_weights_from_class_counts(cw)).to(self._device)

        else:
            raise ValueError("Unknown value passed as output_class_weights_strategy, options are " + str(["uniform", "manual", "ipw", "log_class_balance"]))

        # Update the criterion with the new weights
        for i, weight in enumerate(self.class_weights):
            self._step_metadata.scalars["Loss/Class Weights"][self.output_label_map.getLabels(i)[0]] = weight
        self.criterion.set_class_weights(self.class_weights, normalize=self.hyperparameters["input"]["training_parameters"]["normalize_weights"])

    def _log_metadata(self, logger):
        for name, scalars_collection in self._step_metadata.scalars.items():
            if scalars_collection.is_normalizable() and name in self._step_metadata.normalizations.keys():
                normalization_constant = self._step_metadata.normalizations[name]
                normalization_constant = 1 if normalization_constant == 0 else normalization_constant
                logger.add_scalars(
                    name + " (Normalized)",
                    {
                        k: v / normalization_constant
                        for k, v in scalars_collection.as_dict().items()
                    },
                    self._step_metadata.get_step(),
                )
            else:
                logger.add_scalars(name, scalars_collection.as_dict(), self._step_metadata.get_step())

        for name, scalar_collection in self._step_metadata.scalar.items():
            if self._step_metadata.scalar.is_normalizable(name) and name in self._step_metadata.normalizations.keys():
                normalization_constant = self._step_metadata.normalizations[name]
                normalization_constant = 1 if normalization_constant == 0 else normalization_constant
                logger.add_scalar(
                    name + " (Normalized)",
                    scalar_collection / normalization_constant,
                    self._step_metadata.get_step(),
                )
            else:
                logger.add_scalar(name, scalar_collection, self._step_metadata.get_step())

        for name, image in self._step_metadata.images.items():
            logger.add_image(name, image, self._step_metadata.get_step(), dataformats="HWC")

        for name, quantile_collection in self._step_metadata.quantiles.items():
            logger.add_scalars(
                name,
                {
                    "Quantile " + str(q): np.quantile(quantile_collection, q=q)
                    for i, q in enumerate(self._qs)
                },
                self._step_metadata.get_step(),
            )

    def get_tb_logger(self):
        try:
            for logger in self.trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    self._logger = logger.experiment
                    return self._logger
        except:  # pylint: disable=bare-except
            pass

        if self._logger is None:
            self._logger = TensorBoardLogger("tb_logs", name="debug_log").experiment
        return self._logger

    def mark_images_logged(self, mark=True):
        self._images_logged = mark

    def images_have_been_logged(self):
        return self._images_logged

    def should_log_images(self):
        is_new_step = self._cur_step != self.global_step
        images_requested_on_step = (
            self._log_images_every_n_steps > 0
            and self.global_step % self._log_images_every_n_steps == 0
        )
        return is_new_step and images_requested_on_step

    def on_after_backward(self):
        if self._cur_step != self.global_step:
            if self.should_log_images() and self.images_have_been_logged():
                self._step_metadata.images["Gradient Flow"] = inspect_grad_flow(
                    self.model.named_parameters(), " | Step=" + str(self.global_step)
                )
            self._log_metadata(self.get_tb_logger())
            self._reset_aggregation_step_metadata()
            self._cur_step = self.global_step

    def isNormalizedInput(self):
        return self._normalized_inputs
    
    def get_optimizer(self):

        optimizer_parameters = self._hyperparameters["input"]["training_parameters"]["optimizer_parameters"].copy()
        print(optimizer_parameters)
        optimizer_name = optimizer_parameters.pop("name")

        optimizer_cls = None
        for name in dir(optim):
            if name.lower() == optimizer_name.lower():
                 optimizer_cls = getattr(optim, name)
        if optimizer_cls is None:
            raise ValueError(f"Optimizer '{optimizer_name}' not found.")

        return optimizer_cls(self.model.parameters(), **optimizer_parameters)

    def configure_optimizers(self):
        if isinstance(self.model, BaselineModel):
            raise ValueError("Cannot optimize a model with no parameters.")

        optimizer = self.get_optimizer()
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self._hyperparameters["input"]["validation_parameters"]["validation_reduce_lr_on_plateau_patience"],
            cooldown=self._hyperparameters["input"]["validation_parameters"]["validation_reduce_lr_on_plateau_cooldown"],
            mode=self._hyperparameters["input"]["validation_parameters"]["validation_monitor_mode"],
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": self._hyperparameters["input"]["validation_parameters"][
                    "validation_monitor"
                ],
            },
        }

    def configure_checkpoint(self):
        monitor = self._hyperparameters["input"]["validation_parameters"]["validation_monitor"]
        naming_pattern = str(self.getName()) + "-{epoch:02d}-{step}-{" + monitor + ":.5f}"
        return ModelCheckpoint(
            monitor=monitor,
            save_top_k=self._hyperparameters["input"]["validation_parameters"]["validation_checkpoint_save_top_k"],
            mode=self._hyperparameters["input"]["validation_parameters"]["validation_monitor_mode"],
            filename=naming_pattern,
        )
