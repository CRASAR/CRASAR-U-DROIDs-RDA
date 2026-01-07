import pytorch_lightning as L

from modeling.Models.MaskedUNet.UNet import MaskedUNet, AlignedMaskedUNet


class LitMaskedUNetModel(L.LightningModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        super().__init__()

        self._model = (
            MaskedUNet(
                len(input_channel_map),
                len(output_label_map),
                hyperparameters=hyperparameters,
                input_channel_mask_index=input_channel_map.getIdx("mask"),
                output_channel_background_index=output_label_map.getBackgroundClassIdx(),
            )
            .cpu()
            .to(self._device)
        )

    def get_model(self):
        return self._model

class LitAlignedMaskedUNetModel(L.LightningModule):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        super().__init__()

        self._model = (
            AlignedMaskedUNet(
                len(input_channel_map),
                len(output_label_map),
                hyperparameters=hyperparameters,
                input_channel_mask_index=input_channel_map.getIdx("mask"),
                output_channel_background_index=output_label_map.getBackgroundClassIdx(),
            )
            .cpu()
            .to(self._device)
        )

    def get_model(self):
        return self._model
