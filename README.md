# xt-training
  
## Description

This repo contains utilities for training deep learning models in pytorch, developed by [Xtract AI](https://xtract.ai/).

## Installation

From PyPI:
```bash
pip install xt-training
```

From source:
```bash
git clone https://github.com/XtractTech/xt-training.git
pip install ./xt-training
```

## Usage

See specific help on a class or function using `help`. E.g., `help(Runner)`.

#### Training a model

```python
from xt_training import Runner, metrics
from torch.utils.tensorboard import SummaryWriter

# Here, define a class instances for the required objects
# model = 
# optimizer = 
# scheduler = 
# loss_fn = 

# Define metrics - each of these will be printed for each iteration
# Either per-batch of running-average values can be printed
batch_metrics = {
    'eps': metrics.BatchTimer(),
    'acc': metrics.accuracy,
    'kappa': metrics.kappa
}

# Define tensorboard writer
writer = SummaryWriter()

# Create runner
runner = Runner(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    batch_metrics=batch_metrics,
    device='cuda:0',
    writer=writer
)

# Define dataset and loaders
# dataset = 
# train_loader = 
# val_loader = 

# Train
model.train()
train_loss, train_metrics = runner(train_loader)

# Evaluate
model.eval()
val_loss, val_metrics = runner(val_loader)
```
  
## Data Sources

[descriptions and links to data]
  
## Dependencies/Licensing

[list of dependencies and their licenses, including data]

## References

[list of references]
