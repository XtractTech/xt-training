"""Wrappers for handling object detection tasks with xt-training. Based on the scikit learn wrappers.

To use it on object detection tasks, a few simple modifications are required to the standard
xt-training config format, as shown below:

Usage:

    Define train/val/test datasets as normal.
    Datasets should then each be wrapped in the SKDataset class:

    >>> train_dataset = SKDataset(train_dataset)
    >>> val_dataset = SKDataset(val_dataset)
    >>> test_datasets = {k: SKDataset(v) for k, v in test_datasets.items()}

    Similarly, models are made compatible by wrapping them with the ODInterface class:

    >>> import torchvision
    >>> base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    >>> model = ODInterface(base_model=base_model, device='cuda:0')

    Apart from that, we can set the epochs, the optimizer, and the scheduler as normal 
    Pytorch models.
"""

from collections.abc import Iterable

import torch
from torch import nn


class ODInterface(nn.Module):
    """Class to allow using xt-training for Object Detection."""

    def __init__(self, base_model, device=None):
        """Constructor for ODInterface class.

        Arguments:
            base_model -- A torchvision model.
            device {str} -- Whether to put the tensors on GPU or CPU.
        """
        super().__init__()

        self.base_model = base_model
        self.device = device

    def forward(self, x):
        """Forward method for ODInterface.

        Arguments:
            x {list or tuple} -- List of two torch Tensors, inputs and labels.

        Returns:
            tuple -- The tuple contains the loss calculated by model, the predicted y,
                     and the label y that's moved to the specified device.
        """
        x, y = x

        if self.device:
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
            elif isinstance(x, Iterable):
                x = [x_i.to(self.device) for x_i in x]
            else:
                raise ValueError(f'Unsupport x value: {type(x)}')

            # For object detection, y is a dictionary. Hence the values 
            # should be extracted and send to device and then put back.
            if isinstance(y, torch.Tensor):
                y = y.to(self.device)
            elif isinstance(y, Iterable):
                y_tmp = []
                for y_i in y:
                    if isinstance(y_i, dict):
                        y_i = {k: v.to(self.device) for k, v in y_i.items()}
                    else:
                        y_i.to(self.device)
                    y_tmp.append(y_i)
                y = y_tmp
            else:
                raise ValueError(f'Unsupport y value: {type(y)}')
        
        # The output of Object Detection model is a tuple of (loss, y_pred)
        if self.training:
            # Object Detection training process requires (x, y) as input.
            loss, y_pred = self.base_model(x, y)
            # During training, y_pred is an empty list with len==0.
            # Assign a length to it to make the PooledMean metrics work.
            y_pred = [None]
        else:
            # when not in training mode, loss is an empty dictionary.
            loss, y_pred = self.base_model(x)
        
        return (loss, y_pred), y

    def state_dict(self):
        return self.base_model.state_dict()

    def load_state_dict(self, ckpt):
        self.base_model.load_state_dict(ckpt)
