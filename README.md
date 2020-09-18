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

##### Using xt-training (High Level)
First, you must define a config file with the necessary items.
To generate a template config file, run:
```bash
python -m xt_training template path/to/save/dir
```
To generate template files for nni, add the ```--nni``` flag

Instructions for defining a valid config file can be seen at the top of the config file.

After defining a valid config file, you can train your model by running:
```bash
python -m xt_training train path/to/config.py /path/to/save_dir
```

You can test the model by running
```bash
python -m xt_training test path/to/config.py /path/to/save_dir
```

##### Using functional train (Middle Level)
As of version >=2.0.0, xt-training has functional calls for the train and test functions
This is useful if you want to run other code after training, or want any values/metrics returned after training.
This can be called like so:
```python
from xt_training.utils import functional

# model = 
# train_loader = 
# optimizer = 
# scheduler = 
# loss_fn = 
# metrics = 
# epochs = 
# save_dir = 
def on_exit(test_loaders, runner, save_dir, model):
    # Do what you want after training.
    # As of version >=2.0.0. whatever gets returned here will get returned from the functional call
    return runner, model

runner, model = functional.train(
    save_dir,
    train_loader,
    model,
    optimizer,
    epochs,
    loss_fn,
    val_loader=None,
    test_loaders=None,
    scheduler=scheduler,
    is_batch_scheduler=False, # Whether or not to run scheduler.step() every epoch or every step
    eval_metrics=metrics,
    tokenizer=None,
    on_exit=train_exit,
    use_nni=False
)

# Do something after with runner and/or model...
```

A similar functional call exists for test.

##### Using Runner (Low Level)
If you want a little more control and want to define the trianing code yourself, you can utilize the Runner like so:
```python
from xt_training import Runner, metrics
from torch.utils.tensorboard import SummaryWriter

# Here, define class instances for the required objects
# model = 
# optimizer = 
# scheduler = 
# loss_fn = 

# Define metrics - each of these will be printed for each iteration
# Either per-batch or running-average values can be printed
batch_metrics = {
    'eps': metrics.EPS(),
    'acc': metrics.Accuracy(),
    'kappa': metrics.Kappa(),
    'cm': metrics.ConfusionMatrix()
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
runner(train_loader)
batch_metrics['cm'].print()

# Evaluate
model.eval()
runner(val_loader)
batch_metrics['cm'].print()

# Print training and evaluation history
print(runner)
```

#### Scoring a model

```python
import torch
from xt_training import Runner

# Here, define the model
# model = 
# model.load_state_dict(torch.load(<checkpoint file>))

# Create runner
# (alternatively, can use a fully-specified training runner as in the example above)
runner = Runner(model=model, device='cuda:0')

# Define dataset and loaders
# dataset = 
# test_loader = 

# Score
model.eval()
y_pred, y = runner(test_loader, return_preds=True)
```
