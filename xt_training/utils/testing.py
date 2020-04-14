import os
import shutil
from importlib.util import spec_from_file_location, module_from_spec

import torch
from xt_training import Runner, metrics
from xt_training.utils import _import_config, Tee


def default_exit(config, runner, save_dir):
    pass


def test(args):
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    save_dir = args.save_dir

    if os.path.isdir(config_path):
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

    # Initialize logging
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        try:
            shutil.copy(config_path, f'{save_dir}/config.py')
        except shutil.SameFileError:
            pass
        tee = Tee(os.path.join(save_dir, "test.log"))
        results = {}

    config = _import_config(config_path)

    #  Load definitions
    val_loader = getattr(config, 'val_loader', None)
    test_loaders = getattr(config, 'test_loaders', None)
    model = config.model
    loss_fn = getattr(config, 'loss_fn', lambda *_: torch.tensor(0.))
    eval_metrics = getattr(config, 'eval_metrics', {'eps': metrics.EPS()})
    on_exit = getattr(config, 'test_exit', default_exit)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    model = model.to(device)
    
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Define model runner
    runner = Runner(model, loss_fn, batch_metrics=eval_metrics, device=device)

    if val_loader:
        print('\n\nValidation')
        print('-' * 10)
        if save_dir:
            preds, labels = runner(val_loader, 'valid', return_preds=True)
            results['valid'] = {'preds': preds, 'labels': labels}
        else:
            runner(val_loader, 'valid')

    if test_loaders:
        print('\nTest')
        print('-' * 10)
        for loader_name, loader in test_loaders.items():
            if save_dir:
                preds, labels = runner(loader, loader_name, return_preds=True)
                results[loader_name] = {'preds': preds, 'labels': labels}
            else:
                runner(loader, loader_name)

    on_exit(config, runner, save_dir)

    if save_dir:
        torch.save(results, os.path.join(save_dir, 'results.pt'))
        tee.flush()
        tee.close()
