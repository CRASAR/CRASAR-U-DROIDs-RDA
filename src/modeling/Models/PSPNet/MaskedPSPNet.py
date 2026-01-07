import torch
from torch import nn
import torch.nn.functional as F
from modeling.Models.Maskable import Maskable


class MaskedPSPNet(nn.Module, Maskable):
    def __init__(
        self,
        model,
        n_cls,
        input_channel_mask_index=-1,
        output_channel_background_index=-1,
    ):
        # pylint: disable=duplicate-code
        super().__init__()
        Maskable.__init__(
            self, n_cls, input_channel_mask_index, output_channel_background_index
        )
        self.input_channel_mask_index = input_channel_mask_index
        self.output_channel_background_index = output_channel_background_index
        self.mask_output = (
            input_channel_mask_index >= 0 and output_channel_background_index >= 0
        )

        self.softmax = nn.Softmax(dim=1)

        self.model = model
        self.n_cls = n_cls

        if output_channel_background_index >= 0 or input_channel_mask_index >= 0:
            if not self.mask_output:
                raise ValueError(
                    "Both output_channel_background_index and input_channel_mask_index must be specified when masking."
                )

    def forward(self, x, do_softmax=True):
        # pylint: disable=duplicate-code
        mask = None

        x, mask = x
        preds = self.model(x)

        # interpolating to make predications the expected size...
        preds = F.interpolate(
            preds, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        if do_softmax:
            preds = self.softmax(preds)

        if self.mask_output:
            if mask is None:
                mask = x[:, self.input_channel_mask_index]
            masked_preds = self.mask(preds, mask)

            return masked_preds
        return preds

    def load(self, path, strict):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"], strict=strict,assign=True)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        new_state_dict = {}
        for key, v in state_dict.items():
            k = key.replace("model.", "", 1)

            new_state_dict[k] = v

        super().load_state_dict(new_state_dict, strict, assign)
