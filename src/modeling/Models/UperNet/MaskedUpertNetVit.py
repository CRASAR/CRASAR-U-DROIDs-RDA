import torch
from torch import nn
import torch.nn.functional as F
from modeling.Models.Maskable import Maskable


class MaskedUperNet(nn.Module, Maskable):
    def __init__(
        self,
        encoder,
        decoder,
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

        # pylint: disable=duplicate-code
        self.backbone = encoder
        self.decoder = decoder
        self.n_cls = n_cls

        if output_channel_background_index >= 0 or input_channel_mask_index >= 0:
            if not self.mask_output:
                raise ValueError(
                    "Both output_channel_background_index and input_channel_mask_index must be specified when masking."
                )

    def encode_decode(self, im, input_res=None):
        # pylint: disable=duplicate-code
        if input_res is not None:
            x = self.backbone(im, input_res, return_featuremaps=True)
        else:
            x = self.backbone(im, return_featuremaps=True)
        out = self.decoder(x)

        # interpolating to make predications the expected size...
        masks = F.interpolate(
            out, size=im.shape[-2:], mode="bilinear", align_corners=False
        )
        return masks

    def forward(self, x, do_softmax=True):
        mask = None
        if len(x) > 1:
            # Assign each Pixel a GSD
            x, gsd, mask = x

            gsd_ratio = torch.tensor(gsd, dtype=torch.float32, device=x.device)
            input_res = torch.ones(len(x), device=x.device).float() * gsd_ratio
            # pylint: disable=duplicate-code
            preds = self.encode_decode(x, input_res)
        else:
            preds = self.encode_decode(x)

        # pylint: disable=duplicate-code
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