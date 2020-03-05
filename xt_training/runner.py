import torch
import numpy as np
import json

from collections import Iterable
from .metrics import EPS, PooledMean, Metric

class Logger(object):

    def __init__(self, mode, length):
        """Text logging class.

        Arguments:
            mode {str} -- Run mode, used as a prefix in log output (e.g., 'train' or 'valid').
            length {int} -- Length of training loop, generally the number of batches in an epoch
                (i.e., the length of the dataloader).
        """
        self.mode = mode
        self.length = length

    def __call__(self, loss, metrics, i):
        track_str = '\r{:8s} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
        loss_str = 'loss: {:9.4f} | '.format(loss)
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, v) for k, v in metrics.items())
        print(track_str + loss_str + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')


def detach_objects(x):
    """Function to detach objects from gpu if x is a torch.Tensor or Iterable of torch.Tensors.
    Otherwise, returns the input.
  
    Arguments: 
        x {torch.Tensor or Iterable} -- Object to be detached, can be a torch tensor or an iterable
            of torch Tensors.
  
    Returns: 
        torch.Tensor or Iterable -- If input is a torch.Tensor, returns torch.Tensor. Or if input is
        iterable, returns Iterable. Otherwise, returns the input. 
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, Iterable):
        out = []
        for x_i in x:
            if isinstance(x_i, torch.Tensor):
                out.append(x_i.detach().cpu())
            else:
                out.append(x_i)
        return out
    else:
        return x


class Runner(object):
    """Model trainer/evaluater.

    Switch between training and evaluation modes by setting the model.training attribute with
    model.train() and model.eval().

    Arguments:
        model {nn.Module} -- Model to train/evaluate.
        loss_fn {callable} -- Loss function with signature:
            fn(<model output>, <loader labels>) -> <torch scalar>

    Keyword Arguments:
        optimizer {torch.optim.Optimizer} -- Torch optimizer. Can be None if training will not
            be performed. (default: {None})
        scheduler {torch.optim.lr_scheduler._LRScheduler} -- Torch LR scheduler. Can be None if
            training will not be performed. (default: {None})
        batch_metrics {dict} -- Dict of (named) callables that calculate useful metrics. Each
            should have the same signature as the loss_fn. (default: {{'time': EPS()}})
        device {str or torch.device} -- Device for pytorch to use. (default: {'cpu'})
        writer {torch.utils.tensorboard.SummaryWriter} -- Tensorboard SummaryWriter.
            (default: {None})
        is_batch_scheduler {bool} -- Flag to call scheduler.step every batch instead of every epoch.
            (default: {False})
        logger {object} -- Arbitrary logger object with a call signature that matches the default
            Logger. If None, the default logger will be used. (default: {None})
    """

    def __init__(
        self, model, loss_fn=lambda *_: torch.tensor(0.), optimizer=None, scheduler=None,
        batch_metrics={'eps': EPS()}, device='cpu', writer=None, is_batch_scheduler=False, 
        logger=None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_metrics = batch_metrics
        self.device = device
        self.writer = writer
        self.write_interval = 10
        self.epoch = 0
        self.iteration = 0
        self.history = {}
        self.latest = {}
        self.is_batch_scheduler = is_batch_scheduler
        self.logger = logger

    def __call__(self, loader, mode=None, return_preds=False):
        """Train or evaluate over an epoch of data.

        Arguments:
            loader {torch.utils.data.DataLoader} -- Torch data loader. The loader should return a
                tuple (x, y), where x can be passed directly to the model object (after loading to
                the correct device), and y can be passed directly to the loss and metric functions.

        Keyword Arguments:
            mode {str} -- Prefix for logging (text and tensorboard) (default: {None})
            return_preds {bool} -- Return targets and predictions for all samples in `loader`.
                (default: {False})

        Returns:
            None or tuple -- If `return_preds` is False, returns None. Otherwise, a tuple of the 
                model outputs and the targets.
        """

        assert len(loader) > 0, 'Loader is empty.'

        # Unpack
        model = self.model
        loss_fn = PooledMean(self.loss_fn)
        optimizer = self.optimizer
        scheduler = self.scheduler
        is_batch_scheduler = self.is_batch_scheduler
        batch_metrics = self.batch_metrics
        device = self.device

        # Reset metric cache's where required
        for k in batch_metrics.keys():
            if not isinstance(batch_metrics[k], Metric):
                batch_metrics[k] = PooledMean(batch_metrics[k])
            batch_metrics[k].reset()

        # Set logging prefix if not specified and get logger instance
        if mode is None:
            mode = 'train' if model.training else 'valid'
        if self.logger is None:
            logger = Logger(mode, length=len(loader))
        else:
            logger = self.logger

        if return_preds:
            y_pred_epoch = []
            y_epoch = []

        with torch.set_grad_enabled(model.training):

            for i_batch, (x, y) in enumerate(loader):
                if isinstance(x, torch.Tensor):
                    x = x.to(device)
                elif isinstance(x, Iterable):
                    x = [x_i.to(device) for x_i in x]
                else:
                    raise TypeError('First element returned by loader should be a tensor or list.')

                if isinstance(y, torch.Tensor):
                    y = y.to(device)
                elif isinstance(y, Iterable):
                    y = [y_i.to(device) for y_i in y]
                else:
                    raise TypeError('Second element returned by loader should be a tensor or list.')

                y_pred = model(x)
                loss_batch = loss_fn(y_pred, y)

                if model.training:
                    loss_batch.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None and is_batch_scheduler:
                        scheduler.step()
                    self.iteration += 1

                # Evaluate batch using metrics
                metrics_batch = {nm: fn(y_pred, y) for nm, fn in batch_metrics.items()}
                metrics_batch = {nm: v.detach().cpu() for nm, v in metrics_batch.items() if v is not None}
                metrics = {nm: fn.compute() for nm, fn in batch_metrics.items()}
                metrics = {nm: v.detach().cpu() for nm, v in metrics.items() if v is not None}
                loss = loss_fn.compute()

                if model.training and self.iteration % self.write_interval == 0:
                    self._write(loss_batch, metrics_batch, mode)

                # Log results
                logger(loss, metrics, i_batch)

                if return_preds:
                    y_pred_epoch.append(detach_objects(y_pred))
                    y_epoch.append(detach_objects(y))

        if model.training:
            if not is_batch_scheduler and scheduler is not None:
                scheduler.step()
            self.epoch += 1

        if not model.training:
            self._write(loss, metrics, mode)

        # Save loss and metric values in runner history attribute
        self.history[self.epoch] = self.history.get(self.epoch, {})
        self.history[self.epoch][mode] = {'loss': loss, 'metrics': metrics}
        self.latest = self.history[self.epoch][mode]

        # Combine batches (if feasible)
        if return_preds:
            if all(isinstance(y_i, torch.Tensor) for y_i in y_pred_epoch):
                y_pred_epoch = torch.cat(y_pred_epoch)
            if all(isinstance(y_i, torch.Tensor) for y_i in y_epoch):
                y_epoch = torch.cat(y_epoch)

            return y_pred_epoch, y_epoch

    def __str__(self):
        return (
            'Model training and evaluation runner\n\n'
            'Training and evaluation history:\n'
            '{\n' + ',\n'.join(f'  {k}:{v}' for k, v in self.history.items()) + '\n}'
        )

    def _write(self, loss, metrics, mode):
        if self.writer is None:
            return
        self.writer.add_scalar(f'loss/{mode}', loss.detach().cpu(), self.iteration)
        for metric_name, metric in metrics.items():
            self.writer.add_scalar(f'{metric_name}/{mode}', metric.detach().cpu(), self.iteration)

    def loss(self):
        return self.latest['loss']

    def metrics(self):
        return self.latest['metrics']
