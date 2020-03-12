"""Pytorch deep learning configuration script.

Usage:
>>> python -m xt_training train <path_to_this_file> ...
OR
>>> python -m xt_training test <path_to_this_file> ...

For `python -m xt_training train`:
  Required:
    * train_loader
    * model
    * loss_fn
    * optimizer
    * epochs
  Optional:
    * val_loader
    * test_loaders
    * eval_metrics
    * scheduler
    * train_exit
  Unused:
    * test_exit

For `python -m xt_training test`:
  Required:
    * model
  Optional:
    * loss_fn
    * val_loader
    * test_loaders
    * eval_metrics
    * test_exit
  Unused:
    * train_loader
    * optimizer
    * epochs
    * scheduler
    * train_exit
"""

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.optim import lr_scheduler
from torchvision import models, transforms
import numpy as np
from xt_training import metrics
from xt_training.utils import training, testing

# Dataset
train_dataset = torch.utils.data.TensorDataset(
    torch.rand(128, 3, 224, 224),
    torch.randint(1000, (128,))
)
val_dataset = torch.utils.data.TensorDataset(
    torch.rand(16, 3, 224, 224),
    torch.randint(1000, (16,))
)
test_datasets = {
    'test': torch.utils.data.TensorDataset(
        torch.rand(16, 3, 224, 224),
        torch.randint(1000, (16,))
    )
}

# Dataloaders
batch_size = 8
num_workers = 4
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=RandomSampler(train_dataset)
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=SequentialSampler(val_dataset)
)
test_loaders = {}
for ds_name, ds in test_datasets.items():
    test_loaders[ds_name] = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )

# Model
model = models.resnet18(pretrained=True)

# Loss and metrics
loss_fn = nn.CrossEntropyLoss()
eval_metrics = {
    'acc': metrics.Accuracy(),
    'util': metrics.GPUUtil(),
    'mem': metrics.GPUMem()
}

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Scheduler
epochs = 10
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5])


# Function to run after training
def train_exit(config, runner, save_dir):
    training.default_exit(config, runner, save_dir)


# Function to run after testing
def test_exit(config, runner, save_dir):
    testing.default_exit(config, runner, save_dir)
