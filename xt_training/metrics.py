import torch
import time
from sklearn.metrics import cohen_kappa_score, confusion_matrix as sk_confmat
from functools import lru_cache
import pandas as pd


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
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits)
    if threshold is not None:
        assert probs.shape[1] == 2, "Probability threshold only valid for binary classification"
        preds = (probs[:, 1] >= threshold).long()
    else:
        preds = probs.argmax(dim=1)
    return preds


def _kappa(logits, y):
    preds = logit_to_label(logits)
    return torch.as_tensor(cohen_kappa_score(preds.detach().cpu(), y.detach().cpu()))


def _accuracy(logits, y):
    preds = logit_to_label(logits)
    return (preds == y).float().mean()


def _confusion_matrix(logits, y):
    preds = logit_to_label(logits)
    preds, y = preds.cpu(), y.cpu()
    return torch.as_tensor(sk_confmat(y, preds, labels=[0, 1]))


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
        self.value_sum += self.latest_value.detach().cpu() * self.latest_num_samples
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
        self.latest_num_samples = float(len(y_pred))
        self.latest_eps = self.latest_num_samples / (end - self.start)
        self.eps_sum += self.latest_eps * self.latest_num_samples
        self.num_samples += self.latest_num_samples
        self.start = end

        return torch.tensor(self.latest_eps)

    def compute(self):
        return torch.tensor(self.eps_sum / self.num_samples)

    def reset(self):
        self.start = time.time()
        self.eps_sum = 0
        self.num_samples = 0
        self.latest_eps = 0
        self.latest_num_samples = 0


class Accuracy(PooledMean):
    """Accuracy metric."""

    def __init__(self):
        super().__init__(_accuracy)


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


# Aliases for backward compatibility
accuracy = Accuracy()
kappa = Kappa()
BatchTimer = EPS
