import torch
import time
from sklearn.metrics import cohen_kappa_score
from functools import lru_cache


class BatchTimer(object):

    def __init__(self, rate=True, per_sample=True):
        """Batch timing class.
        
        Use this class for tracking training and testing time/rate per batch or per sample. The
        default parameters are used for calculating examples per second.

        Note that the __call__ method expects two arguments (the predictions and actuals), so that
        its signature matches the loss and metric functions. This means that it can be used in the
        same way.In addition, the timing will be returned as a torch tensor object. 
        
        Keyword Arguments:
            rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
                per batch or sample). (default: {True})
            per_sample {bool} -- Whether to report times or rates per sample or per batch.
                (default: {True})
        
        Example:
        >>> timer = BatchTimer()
        >>> for x, y in loader:
        >>>     y_bar = model(x)
        >>>     loss = loss_fn(y_bar, y)
        >>>     print('EPS: ', timer(y_bar, y))
        """
        self.start = time.time()
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        end = time.time()
        elapsed = end - self.start
        self.start = end

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)


@lru_cache(8)
def logit_to_label(logits, threshold=0.5):
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits)
    preds = (probs[:,1] >= threshold).long()
    return preds


def accuracy(logits, y):
    preds = logit_to_label(logits)
    return (preds == y).float().mean()


def kappa(logits, y):
    preds = logit_to_label(logits)
    return torch.tensor(cohen_kappa_score(preds.detach().cpu(), y.detach().cpu()))
