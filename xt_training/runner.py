import torch
import numpy as np
import json

from .metrics import BatchTimer


class Logger(object):

    def __init__(self, mode, length, calculate_mean=False):
        """Text logging class.
        
        Arguments:
            mode {str} -- Run mode, used as a prefix in log output (e.g., 'train' or 'valid').
            length {int} -- Length of training loop, generally the number of batches in an epoch
                (i.e., the length of the dataloader).
        
        Keyword Arguments:
            calculate_mean {bool} -- Whether to divide values by the iteration count before
                printing. (default: {False})
        """
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x

    def __call__(self, loss, metrics, i):
        track_str = '\r{:8s} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
        loss_str = 'loss: {:9.4f} | '.format(self.fn(loss, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')


class Runner(object):

    def __init__(
        self, model, loss_fn=lambda *_: torch.tensor(0.), optimizer=None, scheduler=None,
        batch_metrics={'eps': BatchTimer()}, show_running=True, device='cpu', writer=None
    ):
        """Model trainer/evaluater.

        Switch between 
        
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
                should have the same signature as the loss_fn. (default: {{'time': BatchTimer()}})
            show_running {bool} -- Whether or not to print losses and metrics for the current batch
                or rolling averages. (default: {False})
            device {str or torch.device} -- Device for pytorch to use. (default: {'cpu'})
            writer {torch.utils.tensorboard.SummaryWriter} -- Tensorboard SummaryWriter.
                (default: {None})
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_metrics = batch_metrics
        self.show_running = show_running
        self.device = device
        self.writer = writer
        self.write_interval = 10
        self.epoch = 0
        self.iteration = 0
        self.history = {}
        self.latest = {}
    
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

        # Unpack
        model = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        scheduler = self.scheduler
        batch_metrics = self.batch_metrics
        show_running = self.show_running
        device = self.device

        # Set logging prefix if not specified and get logger instance
        if mode is None:
            mode = 'train' if model.training else 'valid'
        logger = Logger(mode, length=len(loader), calculate_mean=show_running)
            
        if return_preds:
            y_pred_epoch = []
            y_epoch = []

        loss = 0
        metrics = {}
        for i_batch, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss_batch = loss_fn(y_pred, y)

            if model.training:
                loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.iteration += 1

            # Evaluate batch using metrics
            metrics_batch = {}
            metrics_batch = {nm: fn(y_pred, y).detach().cpu() for nm, fn in batch_metrics.items()}
            metrics = {nm: metrics.get(nm, 0) + metrics_batch[nm] for nm in batch_metrics}
            
            if model.training and self.iteration % self.write_interval == 0:
                self._write(loss_batch, metrics_batch, mode)
            
            # Log results
            loss_batch = loss_batch.detach().cpu()
            loss += loss_batch
            if show_running:
                logger(loss, metrics, i_batch)
            else:
                logger(loss_batch, metrics_batch, i_batch)

            if return_preds:
                y_pred_epoch.append(y_pred.detach().cpu())
                y_epoch.append(y.detach().cpu())
        
        if model.training and scheduler is not None:
            scheduler.step()
            self.epoch += 1

        # Get epoch averages
        loss = (loss / (i_batch + 1)).detach()
        metrics = {k: (v / (i_batch + 1)).detach() for k, v in metrics.items()}

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
