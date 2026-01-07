import os
import pathlib
import copy
import torch

from timm.models.vision_transformer import VisionTransformer
from timm.models.helpers import load_custom_pretrained

from modeling.Models.Backbones.ViT.ScaleMAE import models_vit
from modeling.Models.Backbones.checkpoint_mod import adapt_checkpoint


def load_backbone(backbone_name, backbone_hyperparameters, include_mask=True):

    hyperparmeter_channels = copy.deepcopy(backbone_hyperparameters)
    if not include_mask:
        hyperparmeter_channels["input"]["channels"].pop("mask")

    if backbone_name == "scalemae":
        print("Initializing ViT from ScaleMAE...")
        try:
            output_indicies = backbone_hyperparameters["input"]["model_parameters"][
                "decoder_parameters"
            ]["out_indicies"]
        except KeyError:
            print(
                "Did not find out indicies, assuming decoder does not need, passing in empty list..."
            )
            output_indicies = []

        backbone = models_vit.__dict__[
            backbone_hyperparameters["input"]["model_parameters"]["encoder_parameters"][
                "vit_model"
            ]
        ](
            in_chans=len(hyperparmeter_channels["input"]["channels"]),
            num_classes=backbone_hyperparameters["input"]["model_parameters"]["n_cls"],
            drop_path_rate=backbone_hyperparameters["input"]["model_parameters"][
                "encoder_parameters"
            ]["drop_path_rate"],
            global_pool=backbone_hyperparameters["input"]["model_parameters"][
                "encoder_parameters"
            ]["global_pool"],
            out_indicies=output_indicies,
        )

        if (
            "backbone_path"
            in backbone_hyperparameters["input"]["model_parameters"][
                "encoder_parameters"
            ]
        ):
            print("Found Pretrained Backbone. Loading Pretrained Backbone...")
            platform = os.name

            if platform == "nt":
                # Load the pretrained ViT
                posix_backup = pathlib.PosixPath
                try:
                    pathlib.PosixPath = pathlib.WindowsPath
                    checkpoint = torch.load(
                        backbone_hyperparameters["input"]["model_parameters"][
                            "encoder_parameters"
                        ]["backbone_path"],
                        map_location="cpu",
                    )
                finally:
                    pathlib.PosixPath = posix_backup
            else:
                checkpoint = torch.load(
                    backbone_hyperparameters["input"]["model_parameters"][
                        "encoder_parameters"
                    ]["backbone_path"],
                    map_location="cpu",
                )

            checkpoint_model = checkpoint["model"]
            if len(hyperparmeter_channels["input"]["channels"]) > 3:
                print(
                    "Input Channels Exceed Pretrained 3, initializing extra weights to zero..."
                )
                checkpoint = adapt_checkpoint(checkpoint)
            backbone.load_state_dict(checkpoint_model, strict=False)
        else:
            print(
                "Did not find Pretrained Backbone, weights will be randomly initalized..."
            )
        print("\tDone")
    elif backbone == "vit":
        print("Initializing VisionTransformer with default config for ViT-L-16...")
        default_cfg = dict(
            pretrained=False,
            num_classes=backbone_hyperparameters["input"]["model_parameters"]["n_cls"],
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )
        default_cfg["input_size"] = (
            3,
            backbone_hyperparameters["input"]["training_parameters"]["tile_x"],
            backbone_hyperparameters["input"]["training_parameters"]["tile_y"],
        )
        backbone = VisionTransformer(
            img_size=(
                backbone_hyperparameters["input"]["training_parameters"]["tile_x"],
                backbone_hyperparameters["input"]["training_parameters"]["tile_y"],
            ),
            patch_size=backbone_hyperparameters["input"]["model_parameters"][
                "encoder_parameters"
            ]["patch_size"],
            in_chans=len(hyperparmeter_channels["input"]["channels"]),
            depth=backbone_hyperparameters["input"]["model_parameters"][
                "encoder_parameters"
            ]["n_layers"],
            num_heads=backbone_hyperparameters["input"]["model_parameters"][
                "encoder_parameters"
            ]["n_heads"],
            embed_dim=backbone_hyperparameters["input"]["model_parameters"][
                "encoder_parameters"
            ]["d_model"],
            num_classes=backbone_hyperparameters["input"]["model_parameters"][
                "encoder_parameters"
            ]["n_cls"],
            mlp_ratio=backbone_hyperparameters["input"]["model_parameters"][
                "encoder_parameters"
            ]["mlp_dim"],
        )
        load_custom_pretrained(backbone, default_cfg)
        print("\tDone")
    else:
        raise NotImplementedError()

    return backbone
