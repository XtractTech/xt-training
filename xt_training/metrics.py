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


def _kappa(logits, y, threshold=None):
    preds = logit_to_label(logits, threshold)
    return torch.as_tensor(cohen_kappa_score(preds.detach().cpu(), y.detach().cpu()))


def _accuracy(logits, y, threshold=0.5):
    preds = logit_to_label(logits, threshold)
    return (preds == y).float().mean()


def _topk_accuracy(self, y_pred, y, k):
    topk = torch.topk(y_pred, k, dim=1)[1]
    return (topk == y.unsqueeze(1).repeat(1, self.k)).any(dim=1).float().mean()


def _auc(fpr, tpr):
    """Calculate AUC given FPR and TPR values.
    
    Note that this function assumes the FPR and TPR values are sorted according to monotonically 
    increasing probability thresholds.
    """
    widths = fpr[:-1] - fpr[1:]
    heights = (tpr[:-1] + tpr[1:]) / 2
    return (widths * heights).sum()


@lru_cache(32)
def _crosstab(preds, y, binary=False):
    dev = 'cpu' if preds.get_device() == -1 else preds.get_device()

    if binary:
        correct = preds == y
        y = y.bool()
        cm = torch.zeros(2, 2, device=dev)
        cm[0, 0] = (correct & ~y).sum()
        cm[0, 1] = (~correct & ~y).sum()
        cm[1, 0] = (~correct & y).sum()
        cm[1, 1] = (correct & y).sum()

    else:
        y_unique = torch.unique(torch.cat((preds, y)).int()).sort()[0]
        cm = torch.zeros(y_unique[-1] + 1, y_unique[-1] + 1, device=dev)
        for cls_y in y_unique:
            for cls_yp in y_unique:
                cm[cls_y, cls_yp] = ((y == cls_y) & (preds == cls_yp)).sum()

    return cm


def _confusion_matrix(logits, y, threshold=None):
    preds = logit_to_label(logits, threshold=threshold)
    return _crosstab(preds, y)


def _confusion_matrix_array(logits, y, thresholds, do_softmax=True):
    """For binary classification only - an intermediate step for ROC calculation."""
    dev = 'cpu' if y.get_device() == -1 else y.get_device()
    thresholds = torch.as_tensor(thresholds).to(dev)

    # Get probabilities
    if do_softmax:
        probs = F.softmax(logits, dim=1)[:, 1]
    else:
        probs = logits[:, 1]

    # For efficiency, find all thresholds at which the CM will actually change
    thresh_incr = thresholds[1] - thresholds[0]
    probs_trunc = (probs / thresh_incr).trunc() * thresh_incr
    steps = (thresholds.unsqueeze(1) - probs_trunc.unsqueeze(0)).abs().argmin(dim=0) + 1
    steps = [0] + steps.unique().tolist()

    cm_array = torch.zeros(len(thresholds), 2, 2, device=dev)
    for i, threshold in enumerate(thresholds):
        if i in steps:
            preds = (probs >= float(threshold)).long()
            cm = _crosstab(preds, y, binary=True)
        cm_array[i] = cm

    return cm_array


def _generate_plot(x, y, text, xlabel, ylabel, label, fig, transparent=False):
    if fig is None:
        fig = go.Figure()
        fig.update_layout(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis={'range': [-0.01, 1.005]},
            yaxis={'scaleanchor': "x", 'scaleratio': 1, 'range': [-0.01, 1.0]},
            height=600, width=690,
            margin={'t': 10, 'b': 10, 'l': 10, 'r': 10},
        )
        if transparent:
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#808080'},
                xaxis={'gridcolor': '#CCCCCC', 'zerolinecolor': '#CCCCCC'},
                yaxis={'gridcolor': '#CCCCCC', 'zerolinecolor': '#CCCCCC'},
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

    def __init__(self, threshold=None):
        if threshold is not None and abs(threshold - 0.5) < 1e-5:
            threshold = None
        fn = lambda y_pred, y: _accuracy(y_pred, y, threshold)
        super().__init__(fn)


class TopKAccuracy(PooledMean):
    """Top K Accuracy metric."""

    def __init__(self, k):
        fn = lambda y_pred, y: _accuracy(y_pred, y, k)
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

    def __init__(self, threshold=None, classnames=None):
        self.classnames = classnames
        self.fn = lambda *x: _confusion_matrix(*x, threshold=threshold)
        super().__init__()

    def __call__(self, y_pred, y):
        new_value = self.fn(y_pred, y)

        if len(new_value) > len(self.value) - 1:
            old_value = self.value
            self.value = new_value
            new_value = old_value

        inds = list(range(len(new_value)))

        for ind_i in inds:
            for ind_j in inds:
                self.value[ind_i, ind_j] += new_value[ind_i, ind_j]


    def compute(self):
        pass

    def reset(self):
        self.value = torch.zeros(1, 1)
    
    def print(self):
        mat = self.value.detach().cpu().numpy()
        if self.classnames is None:
            classnames = [str(i) for i in range(len(mat))]
        else:
            classnames = self.classnames
        print(pd.DataFrame(mat, columns=classnames, index=classnames))


class _ConfusionMatrixCurve(Metric):
    """Base class for metrics that utilize confusion matrices (e.g., ROC AUC).
    
    Keyword Arguments:
        increment {float} -- Probability increment. (default: {0.02})
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
        return self._compute_values(self.latest_value.detach())
    
    def compute(self):
        return self._compute_values(self.value_sum)

    def reset(self):
        self.value_sum = 0
        self.num_samples = 0
        self.latest_value = 0
        self.latest_num_samples = 0
    
    def _compute_values(self, cms):
        raise NotImplementedError


class ROC_AUC(_ConfusionMatrixCurve):
    """Metric class to iteratively calculate the ROC curve and AUC.
    
    Keyword Arguments:
        increment {float} -- Probability increment. (default: {0.02})
    """

    def __call__(self, y_pred, y):
        self.latest_value = self.fn(y_pred, y)
        self.latest_num_samples = float(len(y_pred))
        self.num_samples += self.latest_num_samples
        self.value_sum += self.latest_value.detach()
        return torch.as_tensor(self._compute_values(self.latest_value.detach())[0])
    
    def compute(self):
        return torch.as_tensor(self._compute_values(self.value_sum)[0])
    
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

        fpr = cms[:, 0, 1] / (negatives + 1e-6)
        tpr = cms[:, 1, 1] / (positives + 1e-6)

        auc_score = _auc(fpr, tpr)

        return auc_score, fpr, tpr


class BestAccuracy(_ConfusionMatrixCurve):
    """Metric class to iteratively accumulate correct counts for various thresholds and return the
    best possible accuracy.
    
    Keyword Arguments:
        increment {float} -- Probability increment. (default: {0.02})
    """
    
    def _compute_values(self, cms):
        correct = cms[:, 0, 0] + cms[:, 1, 1]
        count = cms[0].sum().float()

        accuracy = correct.float() / count
        accuracy, best_ind = torch.max(accuracy, dim=0)

        self.best_prob = self.probs[best_ind]

        return accuracy


class FPR(_ConfusionMatrixCurve):
    """Metric class to iteratively calculate false positive rate for a given true positive rate.
    
    Keyword Arguments:
        tpr {float} -- Reference true positive rate value. (default: {0.9})
        increment {float} -- Probability increment. (default: {0.02})
    """
    
    def __init__(self, tpr=0.9, increment=0.02):
        self.tpr = tpr
        super().__init__(increment)

    def _compute_values(self, cms):
        negatives = cms[0, 0, 1]
        positives = cms[0, 1, 1]

        if negatives > 1 and positives > 1:
            fpr = cms[:, 0, 1] / negatives
            tpr = cms[:, 1, 1] / positives
        
            ind = torch.nonzero((tpr - self.tpr) >= 1e-6, as_tuple=True)[0][-1]
            
            return fpr[ind]
        else:
            return torch.tensor(float('nan'))


class TPR(_ConfusionMatrixCurve):
    """Metric class to iteratively calculate true positive rate for a given false positive rate.
    
    Keyword Arguments:
        fpr {float} -- Reference false positive rate value. (default: {0.1})
        increment {float} -- Probability increment. (default: {0.02})
    """
    
    def __init__(self, fpr=0.1, increment=0.02):
        self.fpr = fpr
        super().__init__(increment)

    def _compute_values(self, cms):
        negatives = cms[0, 0, 1]
        positives = cms[0, 1, 1]

        if negatives > 1 and positives > 1:
            fpr = cms[:, 0, 1] / negatives
            tpr = cms[:, 1, 1] / positives

            ind = torch.nonzero((fpr - self.fpr) <= 1e-6, as_tuple=True)[0][0]

            return tpr[ind]
        else:
            return torch.tensor(float('nan'))


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
