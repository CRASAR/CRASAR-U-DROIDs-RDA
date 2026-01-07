import time
import numpy as np
import torch

import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from modeling.constants import (
    TRAINING_STEP_METADATA_TIME_INTER_STEP,
    TRAINING_STEP_METADATA_TIME_INTRA_STEP,
    TRAINING_STEP_METADATA_TIME_PREPROCESS,
    TRAINING_STEP_METADATA_TIME_FORWARD,
    TRAINING_STEP_METADATA_TIME_LOSS,
    TRAINING_STEP_METADATA_TIME_LOG,
    TRAINING_STEP_METADATA_TIME_INIT,
)

from modeling.Models.BaseModel import BaseModel
from modeling.Models.OrthoInferenceWrapper import fuse_bda_tiled_inference
from modeling.formatters.plot_metrics_BDA import generate_confusion_matrix_plot
from modeling.utils.decoder_utils import buildings_to_pixel_counts
from modeling.utils.inspection_utils import inspect_image, inspect_labels

class BaseModelBDA(BaseModel):
    def on_predict_epoch_start(self):
        self.predict_step_outputs.clear()
        self.predicted_labels.clear()
        self.model.eval()

    # pylint: disable-next=arguments-differ, unused-argument
    def predict_step(self, batch, batch_idx):
        x = self.format_batched_sample_for_model(batch)
        y_hat_logits = self.model(x, do_softmax=False) # Obtain logits
        y_hat_preds = F.softmax(y_hat_logits, dim=1) # Obtain preds

        for buildings, gsd, y_hat_i in zip(
            batch.getBatchedBuildings(),
            batch.getBatchedGSD(),
            y_hat_preds,
        ):
            if len(buildings) > 0:
                class_preds = buildings_to_pixel_counts(y_hat_i, buildings, 0, 0, label_to_idx_map=self.output_label_map)
                for b in buildings:
                    self.predict_step_outputs[b.getId()].append({"class_preds": class_preds[b.getId()], "gsd": gsd})

    def on_predict_epoch_end(self):
        self.predicted_labels = fuse_bda_tiled_inference(self.predict_step_outputs)

    def get_predicted_labels(self):
        return self.predicted_labels

    # pylint: disable-next=arguments-differ, unused-argument
    def training_step(self, batch, batch_idx):
        start_time = time.time()

        # Set the model to train mode
        self.model.train()

        init_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_INIT] = init_time - start_time

        # Convert the batch into something that could be consumed by the model
        x = self.format_batched_sample_for_model(batch)

        preprocess_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_PREPROCESS] = preprocess_time - init_time

        # Get the class losses from the model, and then release the input from memory since we don't need it
        y_hat = self.model(x, do_softmax=True)
        del x

        forward_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_FORWARD] = forward_time - preprocess_time

        # Compute the criterion loss
        criterion_loss = self.criterion(y_hat, batch.getBatchedLabels()) * self.criterion_scale_factor

        # Compute the L1 Loss
        l1_reg_loss = self.get_l1_loss()

        # Compute the total loss for the model
        # Warning for nan/inf loss values
        if torch.isinf(criterion_loss) or torch.isnan(criterion_loss):
            print("Warning: criterion_loss contains NaN or Inf! Ignoring sample...")
            loss = l1_reg_loss
            criterion_loss = torch.zeros(criterion_loss.shape)
        else:
            loss = l1_reg_loss + criterion_loss

        loss_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_LOSS] = loss_time - forward_time

        # If we need to log out data...
        if self.should_log_images() and not self.images_have_been_logged():
            self._add_batched_images_to_step_metadata(batch.getBatchedRawImagery(), "Image", inspect_image)
            self._add_batched_images_to_step_metadata(batch.getBatchedLabels(), "Label",lambda x: inspect_labels(x, self.idx2color_map))
            self._add_batched_images_to_step_metadata(torch.argmax(y_hat, 1), "Preds", lambda x: inspect_labels(x, self.idx2color_map))
            self.mark_images_logged()

        # Update the dictionaries that contain our tracking data...
        self._log_labels_update_loss(batch.getBatchedLabels())
        self.log_batch_telemetry(batch)
        self._step_metadata.scalar["Loss/Final Loss"] += float(loss.detach().cpu())
        self._step_metadata.scalar["Loss/BDA Criterion Loss"] += float(criterion_loss.mean().detach().cpu())
        self._step_metadata.scalar["Loss/L1 Regularization Loss"] += float(l1_reg_loss.detach().cpu())
        self._step_metadata.normalizations["Loss/Final Loss"] += len(batch)
        self._step_metadata.normalizations["Loss/BDA Criterion Loss"] += len(batch)
        self._step_metadata.normalizations["Loss/L1 Regularization Loss"] += len(batch)

        log_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_LOG] += log_time - loss_time
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_INTER_STEP] += log_time - start_time
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_INTRA_STEP] += start_time - self._prev_start_time
        self._step_metadata.normalizations["Timing/Training Step Timings"] += len(batch)

        self._cur_iter += 1
        self._prev_start_time = start_time

        return loss

    # pylint: disable-next=arguments-differ, unused-argument
    def validation_step(self, batch, batch_idx):
        x = self.format_batched_sample_for_model(batch)
        y_hat = self.model(x)
        del x

        for buildings, gsd, y_hat_i in zip(batch.getBatchedBuildings(), batch.getBatchedGSD(), y_hat):
            if len(buildings) > 0:
                class_preds = buildings_to_pixel_counts(y_hat_i, buildings, 0, 0, label_to_idx_map=self.output_label_map)
                for b in buildings:
                    self.validation_step_outputs[b.getId()].append({"class_preds": class_preds[b.getId()], "gsd": gsd})
                    self.validation_step_labels[b.getId()] = b.getLabel()
                    for c in self.output_label_map.getAllLabels():
                        self._step_metadata.scalars["val/Predicted_Pixel_Counts"][c] += float(class_preds[b.getId()][c])
                self._step_metadata.normalizations["val/Predicted_Pixel_Counts"] += len(buildings)

        # Compute the crtierion loss
        criterion_loss = self.criterion(y_hat, batch.getBatchedLabels()) * self.criterion_scale_factor
        self.validation_loss.append(criterion_loss.detach().cpu().tolist())

    def on_validation_epoch_end(self):
        fused_preds = fuse_bda_tiled_inference(self.validation_step_outputs)
        for c in self.output_label_map.getAllLabels():
            self._step_metadata.scalars["val/Predicted_Class_Counts"][c] = 0
        for pred in fused_preds.values():
            self._step_metadata.scalars["val/Predicted_Class_Counts"][pred["label"]] += 1

        actual_labels = []
        preds_labels = []
        for building_id, pred in self.validation_step_labels.items():
            actual_labels.append(pred)
            preds_labels.append(fused_preds[building_id]["label"])

        macro_f1 = f1_score(actual_labels, preds_labels, average="macro")
        micro_f1 = f1_score(actual_labels, preds_labels, average="micro")

        macro_precision = precision_score(actual_labels, preds_labels, average="macro")
        micro_precision = precision_score(actual_labels, preds_labels, average="micro")

        macro_recall = recall_score(actual_labels, preds_labels, average="macro")
        micro_recall = recall_score(actual_labels, preds_labels, average="micro")

        bda_confusion_matrix = confusion_matrix(y_true=actual_labels, y_pred=preds_labels, labels=list(self.output_label_map.getAllLabels()))
        matrix_data = {
            "Confusion_Matrix": {
                "matrix": bda_confusion_matrix.tolist(),
                "class_labels": self.output_label_map.getAllLabels(),
            }
        }

        np_image = generate_confusion_matrix_plot(
            [{"metrics": matrix_data, "samples": {"total": len(self.validation_step_labels.keys())}, "step": self.global_step}],
            [self.getName()],
            return_np=True,
        )

        self.log("val_macro_f1", macro_f1)

        logger = self.get_tb_logger()
        logger.add_image("val/ConfusionMatrix", np_image, self.global_step, dataformats="HWC")

        logger.add_scalar("val/macro_f1", macro_f1, self.global_step)
        logger.add_scalar("val/micro_f1", micro_f1, self.global_step)

        logger.add_scalar("val/macro_precision", macro_precision, self.global_step)
        logger.add_scalar("val/micro_precision", micro_precision, self.global_step)

        logger.add_scalar("val/macro_recall", macro_recall, self.global_step)
        logger.add_scalar("val/micro_recall", micro_recall, self.global_step)

        logger.add_scalar("val/criterion_loss", np.mean(self.validation_loss), self.global_step)
