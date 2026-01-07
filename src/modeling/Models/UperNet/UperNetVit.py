import os
import torch
import pytorch_lightning as L
from mmseg.registry import MODELS

from modeling.Models.UperNet.MaskedUpertNetVit import MaskedUperNet
from modeling.Models.Backbones.load_backbone import load_backbone


class UperNetVit(L.LightningModule):
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
                type="UPerHead",
                in_channels=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["in_channels"],
                in_index=tuple(
                    hyperparameters["input"]["model_parameters"]["decoder_parameters"][
                        "in_index"
                    ]
                ),
                pool_scales=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["pool_scales"],
                channels=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["channels"],
                dropout_ratio=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["dropout_ratio"],
                num_classes=len(output_label_map),
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                align_corners=hyperparameters["input"]["model_parameters"][
                    "decoder_parameters"
                ]["align_corners"],
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
        self._model = MaskedUperNet(
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
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint["model"])
