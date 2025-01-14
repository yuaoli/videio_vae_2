'''
Author: tom
Date: 2025-01-10 17:34:51
LastEditors: Do not edit
LastEditTime: 2025-01-10 19:05:19
FilePath: /videio_vae/3d-knee-vae/experiments/model/Dynunet.py
'''
from monai.networks.nets import DynUNet
import torch.nn as nn


def get_nnunet_parameters():
    sizes, spacings = (32, 384, 384), (0.700, 0.365, 0.365)
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

def init_dynunet(in_channels,out_channels):

    kernels, strides = get_nnunet_parameters()

    net = DynUNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernels,
                strides=strides,
                filters = [4,32,64,128,256,320],
                upsample_kernel_size=strides[1:],
                norm_name="instance",
            )
    return net
