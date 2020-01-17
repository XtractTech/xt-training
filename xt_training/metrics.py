import torch
from torch.nn import functional as F
import time
from sklearn.metrics import cohen_kappa_score
import plotly.graph_objects as go
from functools import lru_cache
import pandas as pd
import numpy as np

if torch.cuda.is_available():
    from pynvml.smi import nvidia_smi
    smi_instance = nvidia_smi.getInstance()


@lru_cache(8)
def logit_to_label(logits, threshold=None):
    """Convert logits into predicted labels. This function uses a least-recently-used cache to
    avoid recalculating predictions for multiple metric functions.

    Arguments:
        logits {torch.Tensor} -- Tensor of logits of size (batch_size x num_classes).

    Keyword Arguments:
        threshold {float} -- Optional threshold value to use in converting to labels. Only valid
            when num_classes = 2. (default: {None})

    Returns:
        torch.Tensor -- Tensor of predicted labels of length batch_size.
    """
    if threshold is not None:
        assert logits.shape[1] == 2, "Probability threshold only valid for binary classification"
        probs = F.softmax(logits, dim=1)
        preds = (probs[:, 1] >= float(threshold)).long()
    else:
        preds = logits.argmax(dim=1)
    return preds


def _kappa(logits, y, threshold=0.5):
    preds = logit_to_label(logits, threshold)
    return torch.as_tensor(cohen_kappa_score(preds.detach().cpu(), y.detach().cpu()))


def _accuracy(logits, y, threshold=0.5):
    preds = logit_to_label(logits, threshold)
    return (preds == y).float().mean()


def _auc(fpr, tpr):
    """Calculate AUC given FPR and TPR values.
    
    Note that this function assumes the FPR and TPR values are sorted according to monotonically 
    increasing probability thresholds.
    """
    widths = fpr[:-1] - fpr[1:]
    heights = (tpr[:-1] + tpr[1:]) / 2
    return (widths * heights).sum()


@lru_cache(32)
def _crosstab(a, b):
    dev = 'cpu' if a.get_device() == -1 else a.get_device()
    correct = a == b
    b = b.bool()
    cm = torch.zeros(2, 2, device=dev)
    cm[0, 0] = (correct & ~b).sum()
    cm[0, 1] = (~correct & ~b).sum()
    cm[1, 0] = (~correct & b).sum()
    cm[1, 1] = (correct & b).sum()
    return cm


def _confusion_matrix(logits, y, threshold=None):
    preds = logit_to_label(logits, threshold=threshold)
    return _crosstab(preds, y)


def _confusion_matrix_array(logits, y, thresholds, do_softmax=True):
    dev = 'cpu' if y.get_device() == -1 else y.get_device()
    thresholds = torch.as_tensor(thresholds).to(dev)

    # Get probabilities
    if do_softmax:
        probs = F.softmax(logits, dim=1)[:, 1]
    else:
        probs = logits[:, 1]
    y_bool = y.bool()

    # For efficiency, find all thresholds at which the CM will actually change
    thresh_incr = thresholds[1] - thresholds[0]
    probs_trunc = (probs / thresh_incr).trunc() * thresh_incr
    steps = (thresholds.unsqueeze(1) - probs_trunc.unsqueeze(0)).abs().argmin(dim=0) + 1
    steps = [0] + steps.unique().tolist()

    cm_array = torch.zeros(len(thresholds), 2, 2, device=dev)
    for i, threshold in enumerate(thresholds):
        if i in steps:
            preds = (probs >= float(threshold)).long()
            cm = _crosstab(preds, y)
        cm_array[i] = cm

    return cm_array


def _generate_plot(x, y, text, xlabel, ylabel, label, fig):
    if fig is None:
        fig = go.Figure()
        fig.update_layout(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis={'range': [-0.05, 1.05]},
            yaxis={'scaleanchor': "x", 'scaleratio': 1, 'range': [-0.05, 1.05]},
            height=700, width=800
        )

    fig.add_trace(go.Scatter(x=x, y=y, text=text, name=label))
    
    return fig


class Metric(object):
    """Base class for creating metrics that use a cache of passed values (e.g., confusion matrix
    cells).

    Raises:
        NotImplementedError: when __call__ not defined.
        NotImplementedError: when reset not defined.
    """

    def __init__(self):
        """Initialize cached values."""
        self.reset()

    def __call__(self):
        """Update cache and return current (batch-specific) metric value."""
        raise NotImplementedError

    def compute(self):
        """Return overall metric value, combined across all batches."""
        raise NotImplementedError

    def reset(self):
        """Reset cached values."""
        raise NotImplementedError


class PooledMean(Metric):
    """Base class for converting a standard callable into a pooled mean metric.

    This class will store results from calls so that the overall mean can be calculated in 
    MapReduce fashion.

    E.g.,
    First __call__: fn(y_pred, y) -> 10.4, len(y) -> 32
    Second __call__: fn(y_pred, y) -> 3.4, len(y) -> 24
    compute: (10.4 * 32 + 3.4 * 24) / (32 + 24)

    Arguments:
        fn {callable} -- A function that returns a torch scalar metric value. It should have the 
            signature `fn(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor`.
    """

    def __init__(self, fn):
        self.fn = fn
        super().__init__()

    def __call__(self, y_pred, y):
        self.latest_value = self.fn(y_pred, y)
        self.latest_num_samples = float(len(y_pred))
        self.num_samples += self.latest_num_samples
        self.value_sum += self.latest_value.detach() * self.latest_num_samples
        return self.latest_value

    def compute(self):
        return self.value_sum / self.num_samples

    def reset(self):
        self.value_sum = 0
        self.num_samples = 0
        self.latest_value = 0
        self.latest_num_samples = 0


class EPS(Metric):
    """Examples per second.

    Use this class for tracking training and testing samples per second. Note that the __call__
    method expects two arguments (the predictions and actuals), so that its signature matches the
    loss and metric functions. This means that it can be used in the same way. In addition, the
    timing will be returned as a torch tensor object.

    Example:
    >>> timer = EPS()
    >>> for x, y in loader:
    >>>     y_bar = model(x)
    >>>     loss = loss_fn(y_bar, y)
    >>>     print('EPS: ', timer(y_bar, y))
    """

    def __call__(self, y_pred, y):
        end = time.time()
        self.latest_elapsed = end - self.start
        self.latest_num_samples = float(len(y_pred))
        self.elapsed += self.latest_elapsed
        self.num_samples += self.latest_num_samples
        self.start = end

        return torch.tensor(self.latest_num_samples / self.latest_elapsed)

    def compute(self):
        return torch.tensor(self.num_samples / self.elapsed)

    def reset(self):
        self.start = time.time()
        self.elapsed = 0
        self.num_samples = 0
        self.latest_elapsed = 0
        self.latest_num_samples = 0


class Accuracy(PooledMean):
    """Accuracy metric."""

    def __init__(self, threshold=0.5):
        if abs(threshold - 0.5) < 1e-5:
            threshold = None
        fn = lambda y_pred, y: _accuracy(y_pred, y, threshold)
        super().__init__(fn)


class Kappa(PooledMean):
    """Cohen's Kappa metric."""

    def __init__(self):
        super().__init__(_kappa)


class ConfusionMatrix(Metric):
    """Confusion matrix. During training with a runner, this metric will not be logged, but the
    overall confusion matrix can be accessed from the Confusion matrix object later (it is stored
    in the `value` attribute).
    
    Use ConfusionMatrix.print() to print the confusion matrix.
    """

    def __init__(self):
        self.fn = _confusion_matrix
        super().__init__()

    def __call__(self, y_pred, y):
        self.value += self.fn(y_pred, y)

    def compute(self):
        pass

    def reset(self):
        self.value = 0
    
    def print(self):
        print(pd.DataFrame(self.value.numpy(), columns=['N', 'P'], index=['N', 'P']))


class ROC_AUC(Metric):
    """Metric class to iteratively calculate the ROC curve and AUC.
    
    Keyword Arguments:
        probs {torch.Tensor or list} -- Set of probability thresholds to use when building the ROC
            curve. (default: {torch.arange(0, 1.001, 0.01)})
    """

    def __init__(self, increment=0.02):
        self.probs = torch.arange(0, 1+1e-8, increment)
        self.fn = lambda y_pred, y: _confusion_matrix_array(y_pred, y, self.probs)
        super().__init__()

    def __call__(self, y_pred, y):
        self.latest_value = self.fn(y_pred, y)
        self.latest_num_samples = float(len(y_pred))
        self.num_samples += self.latest_num_samples
        self.value_sum += self.latest_value.detach()
        return torch.as_tensor(self._compute_values(self.latest_value.detach())[0])
    
    def compute(self):
        return torch.as_tensor(self._compute_values(self.value_sum)[0])

    def reset(self):
        self.value_sum = 0
        self.num_samples = 0
        self.latest_value = 0
        self.latest_num_samples = 0
    
    def plot(self, fig=None, curve_name=''):
        auc_score, fpr, tpr = self._compute_values(self.value_sum)

        fig = _generate_plot(
            fpr.cpu(), tpr.cpu(), np.array(self.probs).round(3).astype(str),
            'False positive rate', 'True positive rate',
            f"{curve_name} (AUC: {auc_score:.3f})", fig
        )

        return fig
    
    def _compute_values(self, cms):
        negatives = cms[0, 0, 1]
        positives = cms[0, 1, 1]

        fpr = cms[:, 0, 1] / negatives
        tpr = cms[:, 1, 1] / positives

        auc_score = _auc(fpr, tpr)

        return auc_score, fpr, tpr


def get_gpu_util(*unused):
    if torch.cuda.is_available():
        out = smi_instance.DeviceQuery('utilization.gpu')['gpu']
        out = [v['utilization']['gpu_util'] for v in out]
        out = torch.as_tensor(out).float().mean()
    else:
        out = torch.as_tensor(0.)
    return out


def get_gpu_mem(*unused):
    if torch.cuda.is_available():
        out = smi_instance.DeviceQuery('memory.used, memory.total')['gpu']
        out = [v['fb_memory_usage']['used'] / v['fb_memory_usage']['total'] * 100 for v in out]
        out = torch.as_tensor(out).mean()
    else:
        out = torch.as_tensor(0.)
    return out


class GPUUtil(PooledMean):
    """GPU utilization metric."""

    def __init__(self):
        super().__init__(get_gpu_util)


class GPUMem(PooledMean):
    """GPU memory metric."""

    def __init__(self):
        super().__init__(get_gpu_mem)


# Aliases for backward compatibility
accuracy = Accuracy()
kappa = Kappa()
BatchTimer = EPS
