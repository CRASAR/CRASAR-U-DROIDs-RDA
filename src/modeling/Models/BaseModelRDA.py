import time
import numpy as np
import torch

from modeling.constants import (
    TRAINING_STEP_METADATA_TIME_INTER_STEP,
    TRAINING_STEP_METADATA_TIME_INTRA_STEP,
    TRAINING_STEP_METADATA_TIME_PREPROCESS,
    TRAINING_STEP_METADATA_TIME_FORWARD,
    TRAINING_STEP_METADATA_TIME_LOSS,
    TRAINING_STEP_METADATA_TIME_LOG,
    RDA_SAMPLE_METADATA_GENERATING_TIMING,
    TRAINING_STEP_METADATA_TIMING,
    TRAINING_STEP_METADATA_TIME_INIT,
)

from modeling.utils.decoder_utils import road_lines_to_labeled_road_line_segments
from modeling.formatters.plot_metrics_RDA import generate_confusion_matrix_plot
from modeling.Models.BaseModel import BaseModel
from modeling.evaluate_RDA import load_multi_labeled_road_lines_from_preds, compute_metrics, compute_confusion_matrix_for_road_pair
from modeling.Spatial import MultiLabeledRoadLineFactory
from modeling.utils.inspection_utils import inspect_image, inspect_labels
from modeling.utils.sample_generator_utils import translate_road_line

class BaseModelRDA(BaseModel):
    def __init__(self, hyperparameters=None, val_orthomosaics=None, device="cuda"):
        super().__init__(hyperparameters, val_orthomosaics, device)
        self._road_line_buffer_width_pixels = hyperparameters["input"]["road_line_buffer_width_pixels"]
        self._road_line_segment_length_pixels = hyperparameters["input"]["road_line_segment_length_pixels"]


    def on_predict_epoch_start(self):
        self.predict_step_outputs.clear()
        self.predicted_labels.clear()
        self.model.eval()

    # pylint: disable-next=arguments-differ, unused-argument
    def predict_step(self, batch, batch_idx):
        x = self.format_batched_sample_for_model(batch)
        y_hat = self.model(x)

        for road_lines, x_offset, y_offset, y_hat_i in zip(batch.getBatchedRoadLines(), batch.getBatchedX(), batch.getBatchedY(), y_hat):
            if len(road_lines) > 0:
                labeled_road_line_segments = road_lines_to_labeled_road_line_segments(
                    y_hat_i,
                    road_lines,
                    0,
                    0,
                    label_to_idx_map=self.output_label_map,
                    segment_length_pixels=self._road_line_segment_length_pixels,
                    segment_buffer_width_pixels=self._road_line_buffer_width_pixels,
                )
                for parent_road_line, segment_payload in labeled_road_line_segments.items():
                    translated_parent = translate_road_line(parent_road_line, x_offset, y_offset)
                    for labeled_segment in segment_payload["segments"]:
                        translated_labeled_segment = translate_road_line(labeled_segment, -1*x_offset, -1*y_offset)
                        self.predict_step_outputs[translated_labeled_segment.getParentRoadLineId()].append(labeled_segment.jsonify(parent_road_line=translated_parent))

    def on_predict_epoch_end(self):
        self.predicted_labels = self.predict_step_outputs

    def get_predicted_labels(self):
        return self.predicted_labels

    # pylint: disable-next=arguments-differ, unused-argument
    def training_step(self, batch, batch_idx):
        start_time = time.time()

        # Set the model to train mode
        self.model.train()

        init_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_INIT] = init_time - start_time

        # Get the class losses from the model...
        x = self.format_batched_sample_for_model(batch)

        preprocess_time = time.time()
        self._step_metadata.scalars["Timing/Training Step Timings"][TRAINING_STEP_METADATA_TIME_PREPROCESS] = preprocess_time - init_time

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
        self._step_metadata.scalar["Loss/RDA Criterion Loss"] += float(criterion_loss.mean().detach().cpu())
        self._step_metadata.scalar["Loss/L1 Regularization Loss"] += float(l1_reg_loss.detach().cpu())
        self._step_metadata.normalizations["Loss/Final Loss"] += len(batch)
        self._step_metadata.normalizations["Loss/RDA Criterion Loss"] += len(batch)
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

        for road_lines, x_offset, y_offset, y_hat_i in zip(batch.getBatchedRoadLines(), batch.getBatchedX(), batch.getBatchedY(), y_hat):
            if len(road_lines) > 0:
                labeled_road_line_segments = road_lines_to_labeled_road_line_segments(
                    y_hat_i,
                    road_lines,
                    0,
                    0,
                    label_to_idx_map=self.output_label_map,
                    segment_length_pixels=self._road_line_segment_length_pixels,
                    segment_buffer_width_pixels=self._road_line_buffer_width_pixels,
                )

                for parent_road_line, segment_payload in labeled_road_line_segments.items():
                    translated_parent = translate_road_line(parent_road_line, x_offset, y_offset)
                    for labeled_segment in segment_payload["segments"]:
                        translated_labeled_segment = translate_road_line(labeled_segment, -1*x_offset, -1*y_offset)
                        self.validation_step_outputs[translated_labeled_segment.getParentRoadLineId()].append(labeled_segment.jsonify(parent_road_line=translated_parent))
                        self._step_metadata.scalars["val/Predicted_Pixel_Counts"][labeled_segment.getLabel()] += float(labeled_segment.getGeometry("pixels").length)
                self._step_metadata.normalizations["val/Predicted_Pixel_Counts"] += len(road_lines)

        # Compute the criterion loss
        criterion_loss = self.criterion(y_hat, batch.getBatchedLabels()) * self.criterion_scale_factor
        self.validation_loss.append(criterion_loss.mean().cpu().tolist())

    def on_validation_epoch_end(self):
        val_loss = np.mean(self.validation_loss)
        self.validation_step_labels = self.validation_step_outputs
        self.log("val_criterion_loss", val_loss)
        self.get_tb_logger().add_scalar("val/criterion_loss", val_loss, self.current_epoch)

        # Load all of the ground truth multilabeled roadlines
        gt_multilabeled_road_lines = {}
        road_lines_to_gsd = {}
        all_roads = []
        for gt_orthomosaic in self.val_orthomosaics:
            road_lines = gt_orthomosaic.get_road_lines(adjusted=True)
            all_roads.extend(road_lines)
            annotation_polygons = gt_orthomosaic.get_road_line_annotation_polygons()

            road_lines_with_multiple_labels = MultiLabeledRoadLineFactory(road_lines, annotation_polygons)

            for road_line_with_multiple_labels in road_lines_with_multiple_labels:
                gt_multilabeled_road_lines[road_line_with_multiple_labels.getId()] = road_line_with_multiple_labels
                road_lines_to_gsd[road_line_with_multiple_labels.getId()] = gt_orthomosaic.get_gsd()

        pred_multilabeled_road_lines = load_multi_labeled_road_lines_from_preds(self.validation_step_labels, gt_multilabeled_road_lines)

        # Get the labels that are going to be used...
        confusion_matrix_pixels = {}
        confusion_matrix_km = {}

        # Get the labels for the confusion matrix
        conf_matrix_labels = list(self.output_label_map.getAllLabels())
        conf_matrix_labels.remove(self.output_label_map.getBackgroundClass()[0])

        # TODO: Make debug orthos based on the multilabel road lines.
        # TODO: Consider variations in GSD between the x and y axis
        total_km_gsd_calculated = 0
        for line in pred_multilabeled_road_lines:
            total_km_gsd_calculated += (
                (road_lines_to_gsd[line.getId()][0] / 100) / 1000
            ) * line.getGeometry("pixels").length

        for gt_label in conf_matrix_labels:
            confusion_matrix_pixels[gt_label] = {}
            confusion_matrix_km[gt_label] = {}
            for pred_label in conf_matrix_labels:
                confusion_matrix_pixels[gt_label][pred_label] = 0
                confusion_matrix_km[gt_label][pred_label] = 0

        # For every road we have...
        for line in pred_multilabeled_road_lines:
            confusion_matrix_pixels = compute_confusion_matrix_for_road_pair(
                gt_multilabeled_road_lines[line.getId()],
                line,
                confusion_matrix_pixels,
                self.dataset_label_map,
                self.output_label_map,
                default_label=self.default_label,
                scale=line.getGeometry("pixels").length,
                random_baseline=False,
                label_map=self.output_label_map,
            )
            confusion_matrix_km = compute_confusion_matrix_for_road_pair(
                gt_multilabeled_road_lines[line.getId()],
                line,
                confusion_matrix_km,
                self.dataset_label_map,
                self.output_label_map,
                default_label=self.default_label,
                scale=((road_lines_to_gsd[line.getId()][0] / 100) / 1000) * line.getGeometry("pixels").length,
                random_baseline=False,
                label_map=self.output_label_map,
            )

        # Check Here

        confusion_matrix_pixels_list = []
        confusion_matrix_km_list = []
        class_counts_km = {}
        for gt_label in conf_matrix_labels:
            confusion_matrix_pixels_list.append([])
            confusion_matrix_km_list.append([])
            class_counts_km[gt_label] = 0
            for pred_label in conf_matrix_labels:
                confusion_matrix_pixels_list[-1].append(
                    confusion_matrix_pixels[gt_label][pred_label]
                )
                confusion_matrix_km_list[-1].append(
                    confusion_matrix_km[gt_label][pred_label]
                )
                if gt_label == pred_label:
                    class_counts_km[gt_label] += confusion_matrix_km[gt_label][pred_label]

        rda_f1 = {}
        rda_accuracy = {}
        rda_precision = {}
        rda_recall = {}
        rda_iou = {}
        for label in conf_matrix_labels:
            d = compute_metrics(confusion_matrix_pixels, label)
            rda_f1[label] = d["f1"]
            rda_accuracy[label] = d["accuracy"]
            rda_precision[label] = d["precision"]
            rda_recall[label] = d["recall"]
            rda_iou[label] = d["iou"]

        logger = self.get_tb_logger()
        self.log("val_macro_f1", sum(rda_f1.values()) / len(conf_matrix_labels))
        logger.add_scalar(
            "val/macro_f1",
            sum(rda_f1.values()) / len(conf_matrix_labels),
            self.current_epoch,
        )
        logger.add_scalar(
            "val/macro_precision",
            sum(rda_precision.values()) / len(conf_matrix_labels),
            self.current_epoch,
        )
        logger.add_scalar(
            "val/macro_recall",
            sum(rda_recall.values()) / len(conf_matrix_labels),
            self.current_epoch,
        )
        self.log("val_macro_iou", sum(rda_iou.values()) / len(conf_matrix_labels))
        logger.add_scalar(
            "val/macro_iou",
            sum(rda_iou.values()) / len(conf_matrix_labels),
            self.current_epoch,
        )

        metrics= {
            "model_name": self.getName(),
            "samples": {
                "total": total_km_gsd_calculated,
                "class_level": class_counts_km,
            },
            "metrics": {
                "F1": {
                    "class_level": rda_f1,
                    "macro": sum(rda_f1.values()) / len(conf_matrix_labels),
                },
                "Accuracy": {"class_level": rda_accuracy},
                "Precision": {
                    "class_level": rda_precision,
                    "macro": sum(rda_precision.values()) / len(conf_matrix_labels),
                },
                "Recall": {
                    "class_level": rda_recall,
                    "macro": sum(rda_recall.values()) / len(conf_matrix_labels),
                },
                "IoU": {
                    "class_level": rda_iou,
                    "macro": sum(rda_iou.values()) / len(conf_matrix_labels),
                },
                "Confusion_Matrix_pixels": {
                    "matrix": confusion_matrix_pixels_list,
                    "class_labels": conf_matrix_labels,
                },
                "Confusion_Matrix_km": {
                    "matrix": confusion_matrix_km_list,
                    "class_labels": conf_matrix_labels,
                },
            },
        }

        np_image_px = generate_confusion_matrix_plot(
            [metrics], [self.getName()], return_np=True, key="Confusion_Matrix_pixels"
        )
        
        np_image_km = generate_confusion_matrix_plot(
            [metrics], [self.getName()], return_np=True, key="Confusion_Matrix_km"
        )

        logger.add_image(
            "val/ConfusionMatrix_Pixels",
            np_image_px,
            self.current_epoch,
            dataformats="HWC",
        )
        logger.add_image(
            "val/ConfusionMatrix_KM", np_image_km, self.current_epoch, dataformats="HWC"
        )
