import os
import argparse

import datetime

import numpy as np

from dataset.dataset_getter import DatasetGetter
from vision_transformer.models import ViT
from vision_transformer.learner import ViTLearner
from utils.torch import get_device, save_model, load_model
from utils.log import TensorboardLogger
from utils.config import save_yaml, load_from_yaml

import torch

# -----------------------------------------------------------------------------------------------------

device = get_device("cpu")

# Getting Dataset
# dataset = DatasetGetter.get_dataset(
#     dataset_name="cifar100", path="data/", is_train=not False
# )
# dataset_loader = DatasetGetter.get_dataset_loader(
#     dataset=dataset, batch_size=1 if False else 128
# )

# sampled_data = next(iter(dataset_loader))[0].to(device)

# torch.save(sampled_data,'sampled_data.pth')
sampled_data = torch.load('sampled_data.pth')

# n_channel, image_size = sampled_data.size()[1:3]
# -----------------------------------------------------------------------------------------------------

model = ViT(
    image_size=32,
    n_channel=3,
    n_patch=16,
    n_dim=768,
    n_encoder_blocks=12,
    n_heads=12,
    n_classes=100,
    use_cnn_embedding=True,
)

# -----------------------------------------------------------------------------------------------------

# print(model(sampled_data).shape)
print(model(sampled_data))