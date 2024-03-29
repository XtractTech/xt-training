import os
import shutil
import nni
from importlib.util import spec_from_file_location, module_from_spec

import torch
from torch.utils.tensorboard import SummaryWriter
from xt_training import Runner, metrics
from xt_training.utils import _import_config, Tee, functional, _save_config


def train(args):
    config_path = args.config_path
    save_dir = args.save_dir
    overwrite = args.overwrite

    if isinstance(config_path, str):
        config = _import_config(config_path)
    else:
        config = config_path

    #  Load definitions
    train_loader = config.train_loader
    val_loader = getattr(config, "val_loader", None)
    test_loaders = getattr(config, "test_loaders", None)
    model = config.model
    tokenizer = getattr(config, "tokenizer", None)
    optimizer = config.optimizer
    epochs = config.epochs
    scheduler = getattr(config, "scheduler", None)
    is_batch_scheduler = getattr(config, "is_batch_scheduler", False)
    loss_fn = config.loss_fn

    eval_metrics = getattr(config, "eval_metrics", {"eps": metrics.EPS()})
    on_exit = getattr(config, "train_exit", functional.train_exit)
    use_nni = getattr(config, "use_nni", False)

    if use_nni:
        save_dir = os.getenv("NNI_OUTPUT_DIR", save_dir)

    out = functional.train(
        save_dir,
        train_loader,
        model,
        optimizer,
        epochs,
        loss_fn,
        overwrite=overwrite,
        val_loader=val_loader,
        test_loaders=test_loaders,
        scheduler=scheduler,
        is_batch_scheduler=is_batch_scheduler,
        eval_metrics=eval_metrics,
        tokenizer=tokenizer,
        on_exit=on_exit,
        use_nni=use_nni,
    )

    # Save config file in checkpoint directory
    _save_config(save_dir, config_path)

    return out
