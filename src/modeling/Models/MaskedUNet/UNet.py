import torch
from torch import nn

from modeling.Models.MaskedUNet.unet_parts import DoubleConv, Down, AttnUp, Up, OutConv
from modeling.Models.MaskedUNet.alignment_parts import AlignmentDown
from modeling.Models.Maskable import Maskable

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, hyperparameters):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(in_channels=n_channels,
                              out_channels=hyperparameters["input"]["model_parameters"]["layers"]["inc"]["out_channels"],
                              dilation=hyperparameters["input"]["model_parameters"]["layers"]["inc"]["dilation"],
                              kernel_size=hyperparameters["input"]["model_parameters"]["layers"]["inc"]["kernel_size"],
                              padding_mode=hyperparameters["input"]["model_parameters"]["layers"]["inc"]["padding_mode"])

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        last_up_out_channels = None
        for layer in hyperparameters["input"]["model_parameters"]["layers"].keys():
            if "down_" in layer:
                l = Down(in_channels=hyperparameters["input"]["model_parameters"]["layers"][layer]["in_channels"],
                         out_channels=hyperparameters["input"]["model_parameters"]["layers"][layer]["out_channels"],
                         dilation=hyperparameters["input"]["model_parameters"]["layers"][layer]["dilation"],
                         kernel_size=hyperparameters["input"]["model_parameters"]["layers"][layer]["kernel_size"],
                         padding_mode=hyperparameters["input"]["model_parameters"]["layers"][layer]["padding_mode"])
                self.down_layers.append(l)
            if "up_" in layer:
                up_module = AttnUp if hyperparameters["input"]["model_parameters"]["layers"][layer]["attention"] else Up
                factor = 2 if hyperparameters["input"]["model_parameters"]["layers"][layer]["bilinear"] else 1
                l = up_module(in_channels=hyperparameters["input"]["model_parameters"]["layers"][layer]["in_channels"],
                              out_channels=hyperparameters["input"]["model_parameters"]["layers"][layer]["out_channels"] // factor,
                              bilinear=hyperparameters["input"]["model_parameters"]["layers"][layer]["bilinear"])

                last_up_out_channels = hyperparameters["input"]["model_parameters"]["layers"][layer]["out_channels"] // factor
                self.up_layers.append(l)

        #Because this is a unet we assert that there are at least as many down layers as up layers
        assert(len(self.down_layers) == len(self.up_layers))

        self.outc = OutConv(last_up_out_channels, n_classes)

    def _compute_down_convolutions(self, x):
        down_outs = [x]
        for l in self.down_layers:
            down_outs.append(l(down_outs[-1]))
        return down_outs

    def _compute_up_convolutions(self, down_outs):
        intermediate = down_outs[-1]
        for l, d in zip(self.up_layers, down_outs[-2::-1]):
            intermediate = l(intermediate, d)
        return intermediate

    def forward(self, x):
        x_inc = self.inc(x)
        down_outs = self._compute_down_convolutions(x_inc)
        up_convolved_rep = self._compute_up_convolutions(down_outs)
        preds = self.outc(up_convolved_rep)
        return preds

    def use_checkpointing(self):
        # pylint: disable=not-callable
        self.inc = torch.utils.checkpoint(self.inc)
        for i, layer in enumerate(self.down_layers):
            self.down_layers[i] = torch.utils.checkpoint(layer)
        for i, layer in enumerate(self.up_layers):
            self.up_layers[i] = torch.utils.checkpoint(layer)

        self.outc = torch.utils.checkpoint(self.outc)

class MaskedUNet(UNet, Maskable):
    def __init__(self, n_channels, n_classes, hyperparameters, input_channel_mask_index=-1, output_channel_background_index=-1):
        UNet.__init__(self, n_channels, n_classes, hyperparameters)
        Maskable.__init__(self, n_classes, input_channel_mask_index, output_channel_background_index)

        self.softmax = nn.Softmax(dim=1)

        if output_channel_background_index >= 0 or input_channel_mask_index >= 0:
            if not self.mask_output:
                raise ValueError("Both output_channel_background_index and input_channel_mask_index must be specified when masking.")

    def forward(self, x, do_softmax=True):
        preds = super().forward(x)
        if do_softmax:
            preds = self.softmax(preds)
        if self.mask_output:
            mask = x[:, self.input_channel_mask_index]
            return self.mask(preds, mask)
        return preds

    def load(self, path, strict=True, assign=False):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"], strict=strict, assign=assign)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        new_state_dict = {}
        for key, v in state_dict.items():
            k = key.replace("model.", "")

            # For backwards compatiblity
            k = k.replace("down1", "down_layers.0")
            k = k.replace("down2", "down_layers.1")
            k = k.replace("down3", "down_layers.2")
            k = k.replace("down4", "down_layers.3")
            k = k.replace("down5", "down_layers.4")
            k = k.replace("up0", "up_layers.0")
            k = k.replace("up1", "up_layers.1")
            k = k.replace("up2", "up_layers.2")
            k = k.replace("up3", "up_layers.3")
            k = k.replace("up4", "up_layers.4")

            new_state_dict[k] = v

        super().load_state_dict(new_state_dict, strict, assign)

class AlignedMaskedUNet(MaskedUNet):
    def __init__(self, n_channels, n_classes, hyperparameters, input_channel_mask_index=-1, output_channel_background_index=-1):
        super().__init__(n_channels, n_classes, hyperparameters, input_channel_mask_index, output_channel_background_index)
        self.alignment_layer = AlignmentDown(in_channels=self.outc.getInChannels(),
                                             dilation=hyperparameters["input"]["model_parameters"]["layers"]["adj"]["dilation"],
                                             kernel_size=hyperparameters["input"]["model_parameters"]["layers"]["adj"]["kernel_size"],
                                             padding_mode=hyperparameters["input"]["model_parameters"]["layers"]["adj"]["padding_mode"])

        stride_x = int(hyperparameters["input"]["training_parameters"]["mask_x"]/hyperparameters["input"]["training_parameters"]["alignment_map"]["x_dim"])
        stride_y = int(hyperparameters["input"]["training_parameters"]["mask_y"]/hyperparameters["input"]["training_parameters"]["alignment_map"]["y_dim"])
        self.alignment_finalization = nn.AvgPool2d(kernel_size=(stride_x, stride_y), stride=(stride_x, stride_y))

    def use_checkpointing(self):
        # pylint: disable=not-callable
        super().use_checkpointing()
        self.alignment_layer = torch.utils.checkpoint(self.alignment_layer)

    def forward(self, x, do_softmax=True):
        #Consume the input
        x_inc = self.inc(x)

        #Compute the down convolutions in the network
        down_outs = self._compute_down_convolutions(x_inc)

        #Perform the up convolutions
        up_convolved_rep = self._compute_up_convolutions(down_outs)

        #Compute the prediction outputs for each channel
        bda_preds = self.outc(up_convolved_rep)
        if do_softmax:
            bda_preds = self.softmax(bda_preds)

        #Downsample the input and then pass it through an initial layer
        alignment_vector_field = self.alignment_layer(up_convolved_rep)
        downsampled_vector_field = self.alignment_finalization(alignment_vector_field)
        return bda_preds, downsampled_vector_field
