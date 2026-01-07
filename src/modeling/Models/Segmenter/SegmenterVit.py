import torch
import pytorch_lightning as L
from mmseg.registry import MODELS

from modeling.Models.Segmenter.MaskedSegmenterVit import MaskedSegmenterVit
from modeling.Models.Backbones.load_backbone import load_backbone


class SegmenterVit(L.LightningModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        super().__init__()

        # Initialize Encoder
        backbone = load_backbone(
            hyperparameters["input"]["model_parameters"]["encoder_parameters"][
                "backbone"
            ],
            hyperparameters,
            hyperparameters["input"]["mask_input"],
        )

        # Initialize UperNet Decoder
        decoder = MODELS.build(
            dict(
                type="SegmenterMaskTransformerHead",
                in_channels=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["in_channels"],
                channels=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["channels"],
                num_classes=len(output_label_map),
                num_layers=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["num_layers"],
                num_heads=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["num_heads"],
                embed_dims=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["embed_dims"],
                dropout_ratio=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["dropout_ratio"],
            )
        )

        try:
            if hyperparameters["input"]["model_parameters"]["encoder_parameters"][
                "freeze_backbone"
            ]:
                print(
                    "NOTICE: Freezing Weights for backbone, this assumes pretrained backbone..\n"
                )
                for param in backbone.parameters():
                    param.requires_grad = False
        except KeyError:
            pass

        try:
            input_channel_mask_index = input_channel_map.getIdx("mask")
        except KeyError:
            print("Warning: There is no mask channel....Continuing without it...")
            input_channel_mask_index = -1

        # Initalize MaskedUperNet with backbone
        self._model = MaskedSegmenterVit(
            backbone,
            decoder,
            len(output_label_map),
            input_channel_mask_index=input_channel_mask_index,
            output_channel_background_index=output_label_map.getBackgroundClassIdx(),
        )

        # Clamping Gradients to avoid vanishing gradients/too small gradients
        for param in self._model.parameters():
            if param.grad is not None:
                param.grad.data.clamp(min=1e-6)

    def get_model(self):
        return self._model

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
