import torch

class Maskable:
    def __init__(self, n_classes, input_channel_mask_index, output_channel_background_index):
        self.n_maskable_classes = n_classes
        self.input_channel_mask_index = input_channel_mask_index
        self.output_channel_background_index = output_channel_background_index
        self.mask_output = input_channel_mask_index >= 0 and output_channel_background_index >= 0

    def mask(self, softmaxed_preds, mask):
        mask_inv = 1-mask
        stack = [mask]*self.n_maskable_classes
        stack[self.output_channel_background_index] = mask_inv
        stacked_mask = torch.stack(stack, dim=1)
        masked_preds = softmaxed_preds.multiply(stacked_mask)
        return masked_preds
    def get_classes_count(self):
        return self.n_maskable_classes
    def get_input_channel_mask_index(self):
        return self.input_channel_mask_index
    def get_output_channel_background_index(self):
        return self.output_channel_background_index
