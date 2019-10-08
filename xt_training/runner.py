import torch
import numpy as np

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
        self, model, loss_fn=lambda *_: 0, optimizer=None, scheduler=None,
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
    
    def __call__(self, loader, mode=None):
        """Train or evaluate over an epoch of data.
        
        Arguments:
            loader {torch.utils.data.DataLoader} -- Torch data loader. The loader should return a
                tuple (x, y), where x can be passed directly to the model object (after loading to
                the correct device), and y can be passed directly to the loss and metric functions.
        
        Keyword Arguments:
            mode {str} -- Prefix for logging (text and tensorboard) (default: {None})
        
        Returns:
            tuple -- The average loss and metric values for the epoch.
        """
        loss, metrics = _pass_epoch(
            self.model, self.loss_fn, loader, optimizer=self.optimizer,
            scheduler=self.scheduler, batch_metrics=self.batch_metrics,
            show_running=self.show_running, device=self.device, writer=self.writer,
            mode=mode
        )

        return loss, metrics
    
    def score(self, loader):
        return _score(self.model, loader, show_running=self.show_running, device=self.device)


def _pass_epoch(
    model, loss_fn, loader, optimizer=None, scheduler=None,
    batch_metrics={'eps': BatchTimer()}, show_running=True,
    device='cpu', writer=None, mode=None
):
    """Train or evaluate over a data epoch."""
    
    # Set logging prefix if not specified and get logger instance
    if mode is None:
        mode = 'Train' if model.training else 'Valid'
    logger = Logger(mode, length=len(loader), calculate_mean=show_running)

    if writer is not None:
        # Add iteration and interval trackers to writer object
        if not hasattr(writer, 'iteration'):
            writer.iteration = 0
        if not hasattr(writer, 'interval'):
            writer.interval = 10

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

        # Evaluate batch using metrics
        metrics_batch = {}
        metrics_batch = {nm: fn(y_pred, y).detach().cpu() for nm, fn in batch_metrics.items()}
        metrics = {nm: metrics.get(nm, 0) + metrics_batch[nm] for nm in batch_metrics}
        
        if writer is not None and model.training:
            # Write to tensorboard (during training only)
            if writer.iteration % writer.interval == 0:
                writer.add_scalar(f'loss/{mode}', loss_batch.detach().cpu(), writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalar(f'{metric_name}/{mode}', metric_batch, writer.iteration)
            writer.iteration += 1
        
        # Log results
        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
        if show_running:
            logger(loss, metrics, i_batch)
        else:
            logger(loss_batch, metrics_batch, i_batch)
    
    if model.training and scheduler is not None:
        scheduler.step()

    # Get epoch averages
    loss = loss / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}

    if writer is not None and not model.training:
        # Write to tensorboard (during eval only)
        writer.add_scalar(f'loss/{mode}', loss.detach(), writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalar(f'{metric_name}/{mode}', metric, writer.iteration)

    return loss, metrics
    

def _score(model, loader, show_running=True, device='cpu'):
    """Score model against loader and return predictions."""

    # Get logger instance
    batch_metrics = {'eps': BatchTimer()}
    logger = Logger('score', length=len(loader), calculate_mean=show_running)

    loss = 0
    metrics = {}
    y_preds = []
    ys = []
    for i_batch, (x, y) in enumerate(loader):
        x = x.to(device)
        y_pred = model(x)

        # Evaluate batch using metrics
        metrics_batch = {}
        metrics_batch = {nm: fn(y_pred, y).detach().cpu() for nm, fn in batch_metrics.items()}
        metrics = {nm: metrics.get(nm, 0) + metrics_batch[nm] for nm in batch_metrics}

        # Log results
        if show_running:
            logger(0, metrics, i_batch)
        else:
            logger(0, metrics_batch, i_batch)

        y_preds.append(y_pred.detach().cpu())
        ys.append(y)

    # Combine batches (if feasible)
    if all(isinstance(y_i, torch.Tensor) for y_i in y_preds):
        y_preds = torch.cat(y_preds)
    if all(isinstance(y_i, torch.Tensor) for y_i in ys):
        ys = torch.cat(ys)

    # Get epoch averages
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}

    return y_preds, ys
