import torch
import pytorch_lightning as L
from mmseg.models import build_segmentor

from modeling.Models.PSPNet.MaskedPSPNet import MaskedPSPNet


class PSPNetResNet(L.LightningModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        super().__init__()

        # pylint: disable=duplicate-code
        model = build_segmentor(
            dict(
                type="EncoderDecoder",
                backbone=dict(
                    type="ResNetV1c",
                    depth=hyperparameters["input"]["model_parameters"][
                        "encoder_parameters"
                    ]["depth"],
                    num_stages=hyperparameters["input"]["model_parameters"][
                        "encoder_parameters"
                    ]["num_stages"],
                    out_indices=hyperparameters["input"]["model_parameters"][
                        "encoder_parameters"
                    ]["out_indices"],
                    dilations=hyperparameters["input"]["model_parameters"][
                        "encoder_parameters"
                    ]["dilations"],
                    strides=hyperparameters["input"]["model_parameters"][
                        "encoder_parameters"
                    ]["strides"],
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    norm_eval=hyperparameters["input"]["model_parameters"][
                        "encoder_parameters"
                    ]["norm_eval"],
                    style="pytorch",
                    contract_dilation=hyperparameters["input"]["model_parameters"][
                        "encoder_parameters"
                    ]["contract_dilation"],
                ),
                decode_head=dict(
                    type="PSPHead",
                    in_channels=hyperparameters["input"]["model_parameters"][
                        "decoder_parameters"
                    ]["in_channels"],
                    in_index=hyperparameters["input"]["model_parameters"][
                        "decoder_parameters"
                    ]["in_index"],
                    channels=hyperparameters["input"]["model_parameters"][
                        "decoder_parameters"
                    ]["channels"],
                    pool_scales=hyperparameters["input"]["model_parameters"][
                        "decoder_parameters"
                    ]["pool_scales"],
                    dropout_ratio=hyperparameters["input"]["model_parameters"][
                        "decoder_parameters"
                    ]["dropout_ratio"],
                    num_classes=len(output_label_map),
                    #norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    align_corners=hyperparameters["input"]["model_parameters"][
                        "decoder_parameters"
                    ]["align_corners"],
                ),
                auxiliary_head=dict(
                    type="FCNHead",
                    in_channels=hyperparameters["input"]["model_parameters"][
                        "auxiliary_head_parameters"
                    ]["in_channels"],
                    in_index=hyperparameters["input"]["model_parameters"][
                        "auxiliary_head_parameters"
                    ]["in_index"],
                    channels=hyperparameters["input"]["model_parameters"][
                        "auxiliary_head_parameters"
                    ]["channels"],
                    num_convs=hyperparameters["input"]["model_parameters"][
                        "auxiliary_head_parameters"
                    ]["num_convs"],
                    concat_input=hyperparameters["input"]["model_parameters"][
                        "auxiliary_head_parameters"
                    ]["concat_input"],
                    dropout_ratio=hyperparameters["input"]["model_parameters"][
                        "auxiliary_head_parameters"
                    ]["dropout_ratio"],
                    num_classes=len(output_label_map),
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    align_corners=hyperparameters["input"]["model_parameters"][
                        "auxiliary_head_parameters"
                    ]["align_corners"],
                ),
            )
        )
        try:
            input_channel_mask_index = input_channel_map.getIdx("mask")
        except KeyError:
            print("Warning: There is no mask channel....Continuing without it...")
            input_channel_mask_index = -1

        # Initalize PSPNetResNet with backbone
        self._model = MaskedPSPNet(
            model,
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
