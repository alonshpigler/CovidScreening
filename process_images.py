import logging
import pickle
import random
import sys
import time
import scipy
from argparse import ArgumentParser
import numpy as np
import os
from pathlib import Path

import torch
import pytorch_lightning as pl

import config
from data_layer.dataset import CovidDataset
from data_layer.prepare_data import load_data
from model_layer.DummyAutoEncoder import LitAutoEncoder
from model_layer.UNET import Unet
from visuals.util import show_input_and_target

# print('__Python VERSION:', sys.version)
# print('__pyTorch VERSION:', torch.__version__)
# print('__CUDA VERSION')
# print('__CUDNN VERSION:', torch.backends.cudnn.version())
# print('__Number CUDA Devices:', torch.cuda.device_count())
# print('__Devices')
# # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
# # print('Active CUDA Device: GPU', torch.cuda.current_device())
# # print('Available devices ', torch.cuda.device_count())
# # print('Current cuda device ', torch.cuda.current_device())
# use_cuda = torch.cuda.is_available()
# print("USE CUDA=" + str(use_cuda))
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# Tensor = FloatTensor


def process_image(model, input, input_size):
    # based on https://discuss.pytorch.org/t/creating-nonoverlapping-patches-from-3d-data-and-reshape-them-back-to-the-image/51210/6

    # divide to patches
    h, w = input_size, input_size
    input_c, output_c = 4, 1

    input_patches = input.unfold(2, h, w).unfold(3, h, w)  # to patches
    unfold_shape = list(input_patches.shape)
    input_patches = input_patches.permute(0, 3, 2, 1, 4, 5).contiguous().view(-1, input_c, h, w)  # reshape for model

    pred_patches = model.forward(input_patches.to(model.device))  # inference
    # show_input_and_target(input_patches[0,:,:,:].cpu().detach().numpy().squeeze(),pred=pred_patches[0,:,:,:].cpu().detach().numpy().squeeze())
    pred_unfold_shape = unfold_shape.copy()
    pred_unfold_shape[1] = output_c  # mapped from 4 to 1

    # Reshape back
    patches_orig = input_patches.view(unfold_shape)
    pred_orig = pred_patches.view(pred_unfold_shape)
    output_h = unfold_shape[2] * unfold_shape[4]
    output_w = unfold_shape[3] * unfold_shape[5]

    patches_orig = patches_orig.permute(0, 3, 2, 4, 1, 5).contiguous()
    pred_orig = pred_orig.permute(0, 1, 3, 4, 2, 5).contiguous()
    patches_orig = patches_orig.view(1, 4, output_h, output_w).numpy().squeeze()
    pred = pred_orig.view(1, 1, output_h, output_w).cpu().detach().numpy().squeeze()

    # new_pred = pred_orig.permute(0, 1, 2, 3, 4, 5).contiguous()
    # new_pred = new_pred.view(1, 1, output_h, output_w).cpu().detach().numpy().squeeze()
    # show_input_and_target(input[0, :, :, :].cpu().detach().numpy().squeeze(), new_pred)
    # Check for equality
    # assert ((patches_orig == input[:, :output_h, :output_w]).all(),'error in division to patches in inference')

    return pred

