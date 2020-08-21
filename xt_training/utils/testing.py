import os
import shutil
from importlib.util import spec_from_file_location, module_from_spec

import torch
from xt_training import Runner, metrics
from xt_training.utils import _import_config, Tee, functional


def test(args):
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    save_dir = args.save_dir
        
    if isinstance(config_path, str) and os.path.isdir(config_path):
        config_dir = config_path
        config_path = os.path.join(config_dir, 'config.py')
        assert checkpoint_path is None, (
            "checkpoint_path is not valid when config_path is a directory.\n"
            "\tSpecify either a config script and (optional) checkpoint individually, or\n"
            "\tspecify a directory containing both config.py and best.pt"
        )
        checkpoint_path = os.path.join(config_dir, 'best.pt')
        if not save_dir:
            save_dir = config_dir
    

    if isinstance(config_path, str):
        config = _import_config(config_path)
    else:
        config = config_path

    #  Load definitions
    val_loader = getattr(config, 'val_loader', None)
    test_loaders = getattr(config, 'test_loaders', None)
    model = config.model
    loss_fn = getattr(config, 'loss_fn', lambda *_: torch.tensor(0.))
    eval_metrics = getattr(config, 'eval_metrics', {'eps': metrics.EPS()})
    on_exit = getattr(config, 'test_exit', functional.test_exit)

    out = functional.test(
        save_dir,
        model,
        checkpoint_path,
        val_loader,
        test_loaders,
        loss_fn,
        eval_metrics,
        on_exit
    )

    # Save config file
    try:
        if isinstance(config_path, str):
            shutil.copy(config_path, f'{save_dir}/config.py')
        else:
            shutil.copy(config_path['__file__'], f'{save_dir}/config.py')
    except shutil.SameFileError:
        pass

    return out
