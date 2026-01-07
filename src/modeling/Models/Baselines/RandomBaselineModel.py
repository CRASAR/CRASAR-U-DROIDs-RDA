import torch
from torch import nn

from modeling.Models.Baselines.Baseline import BaselineModel

class MaskedRandomModel(nn.Module):
    def __init__(self, n_classes, input_channel_mask_index=-1, output_channel_background_index=-1, class_weights=None):
        super().__init__()
        self.n_classes = n_classes
        self.output_channel_background_index = output_channel_background_index
        self.input_channel_mask_index = input_channel_mask_index
        self.softmax = nn.Softmax(dim=1)
        self.mask_output = output_channel_background_index >= 0
        if class_weights is None:
            self.class_weights = torch.tensor([1.0]*n_classes)
        else:
            self.class_weights = torch.tensor(class_weights)

    def _mask(self, softmaxed_preds, mask):
        mask_inv = 1-mask
        stack = [mask]*self.n_classes
        stack[self.output_channel_background_index] = mask_inv
        stacked_mask = torch.stack(stack, dim=1)
        masked_preds = softmaxed_preds.multiply(stacked_mask)
        return masked_preds

    def forward(self, x):
        preds = torch.rand(x.shape[0], self.n_classes, x.shape[2], x.shape[3])
        softmaxed_preds = self.softmax(preds)
        softmaxed_preds_weighted = self.class_weights[None, :, None, None] * softmaxed_preds
        if self.mask_output:
            mask = x[:, self.input_channel_mask_index]
            return self._mask(softmaxed_preds_weighted.to(mask.device), mask)
        return softmaxed_preds_weighted

    def load(self, path):
        pass

class RandomBaselineModel(BaselineModel):
    def __init__(
        self, hyperparameters=None, input_channel_map=None, output_label_map=None
    ):
        super().__init__()

        #Compute the weights
        weights = [1.0]*len(output_label_map)
        for label in output_label_map.getAllLabels():
            weights[output_label_map.getIndex(label)] = hyperparameters["input"]["class_weights"][label]

        #Initialize the model
        self._model = MaskedRandomModel(n_classes=len(output_label_map),
                                        input_channel_mask_index=input_channel_map.getIdx("mask"),
                                        output_channel_background_index=output_label_map.getBackgroundClassIdx(),
                                        class_weights=weights)

    def load(self, path):
        pass
