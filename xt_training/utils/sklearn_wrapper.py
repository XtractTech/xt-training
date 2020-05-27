"""Wrappers for training/evaluating scikit learn models with xt-training.

To use scikit learn models, a few simple modifications are required to the standard
xt-training config format, as shown below:

Usage:

    Define train/val/test datasets as normal, where each sample returned should typically
    be a row in a tabular dataset (this is generally the format suitable for sklearn).
    Datasets should then each be wrapped in the SKDataset class:

    >>> train_dataset = SKDataset(train_dataset)
    >>> val_dataset = SKDataset(val_dataset)
    >>> test_datasets = {k: SKDataset(v) for k, v in test_datasets.items()}

    Similarly, sklearn models are made compatible by wrapping them with the SKInterface class:

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier()
    >>> model = SKInterface(model, output_dim=2, partial_fit=False)

    Typically, sklearn models are fit to the entire dataset in one go, so we set batch_size and
    epochs accordingly:

    >>> epochs = 1
    >>> batch_size = int(1e9)

    Note that some sklearn models enable iterative fitting via the partial_fit method. In these
    cases, set partial_fit=True when creating the SKInterface object, and you can also use more
    than 1 epoch and a smaller batch_size.

    Finally, since a pytorch optimizer is not required for training sklearn models, but it is
    required by the xt-training `train` utility, a dummy optimizer is provided:

    >>> optimizer = DummyOptimizer()

    A scheduler should not be specified in the config.
"""

import torch
from torch import nn
from sklearn.utils.validation import check_is_fitted, NotFittedError


class SKDataset:
    """Dataset wrapper class to enable training scikit learn models with xt-training."""

    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, i):
        """Get a dataset sample and label

        This returns (x, y), y to ensure that both x and y are passed to the sklearn model's
        forward function, and are hence available for the fit() method.
        """
        x, y = self.dataset[i]
        return (x, y), y
    
    def __len__(self):
        return len(self.dataset)


class SKInterface(nn.Module):
    """Class to allow using Scikit learn models with xt-training."""

    def __init__(self, base_model, output_dim, partial_fit=False):
        """Constructor for SKInterface class.

        Arguments:
            base_model {sklearn.base.BaseEstimator} -- A scikit learn model.
            output_dim {int} -- The intended dimension of the model output. 

        Keyword Arguments:
            partial_fit {bool} -- Whether to use the partial_fit() method instead of fit().
                (default: {False})
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.base_model = base_model
        self.partial_fit = partial_fit
        self.istraining = False
        self.eval()

    def forward(self, x):
        """Forward method for scikit learn interface.

        In train mode, this method will call fit() and then predict_proba() followed by log().

        Arguments:
            x {list or tuple} -- List of two torch Tensors, inputs and labels.

        Returns:
            torch.Tensor -- The output log probabilities.
        """
        x, y = x
        dev = x.device
        x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()

        if self.istraining:
            if self.partial_fit:
                self.base_model.partial_fit(x, y)
            else:
                self.base_model.fit(x, y)

        try:
            check_is_fitted(self.base_model)
            output = torch.as_tensor(self.base_model.predict_proba(x)).to(dev).log()
            return output
        except NotFittedError:
            return torch.zeros(x.shape[0], self.output_dim, device=dev)

    def train(self, mode=True):
        """Training/eval mode setting method.

        This ensures that the model is always in "eval" mode as far as pytorch is concerned. This
        makes sure that the backward pass and optimizer step are skipped by xt-training, since
        they are not relevant for scikit learn models.

        Keyword Arguments:
            mode {bool} -- Whether to set to training mode. (default: {True})

        Returns:
            torch.nn.Module -- The object of this class on which the call was made.
        """
        for module in self.children():
            module.train(False)
        self.training = False
        self.istraining = mode
        return self

    def eval(self):
        return self.train(False)


class DummyOptimizer:
    """A dummy optimizer class to use with the SKInterface in xt-training."""

    def zero_grad(self):
        pass
    
    def step(self):
        pass
