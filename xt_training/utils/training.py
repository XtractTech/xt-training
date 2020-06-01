import os
import shutil
import nni
from importlib.util import spec_from_file_location, module_from_spec

import torch
from torch.utils.tensorboard import SummaryWriter
from xt_training import Runner, metrics
from xt_training.utils import _import_config, Tee


def default_exit(config, runner, save_dir):
    test_loaders = getattr(config, 'test_loaders', None)

    # Final evaluation against test set(s)
    if test_loaders:
        print('\nTest')
        print('-' * 10)
        config.model.eval()
        for loader_name, loader in test_loaders.items():
            runner(loader, loader_name)


def train(args):
    config_path = args.config_path
    save_dir = os.getenv("NNI_OUTPUT_DIR", args.save_dir)
    overwrite = args.overwrite

    # Initialize logging
    if overwrite and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    tee = Tee(os.path.join(save_dir, "train.log"))

    if isinstance(config_path, str):
        config = _import_config(config_path)
        shutil.copy(config_path, f'{save_dir}/config.py')
    else:
        config = config_path
        shutil.copy(config['__file__'], f'{save_dir}/config.py')

    #  Load definitions
    train_loader = config.train_loader
    val_loader = getattr(config, 'val_loader', None)
    test_loaders = getattr(config, 'test_loaders', None)
    model = config.model
    optimizer = config.optimizer
    epochs = config.epochs
    scheduler = getattr(config, 'scheduler', None)
    loss_fn = config.loss_fn
    eval_metrics = getattr(config, 'eval_metrics', {'eps': metrics.EPS()})
    on_exit = getattr(config, 'train_exit', default_exit)
    use_nni = getattr(config, 'use_nni', False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    model = model.to(device)

    # Create tensorboard writer
    writer = SummaryWriter(save_dir, flush_secs=30)

    # Define model runner
    runner = Runner(
        model, loss_fn, optimizer, scheduler, batch_metrics=eval_metrics,
        device=device, writer=writer
    )

    if test_loaders:
        print('\n\nInitial')
        print('-' * 10)
        model.eval()
        for loader_name, loader in test_loaders.items():
            runner(loader, loader_name)

    best_loss = 1e12
    if val_loader:
        model.eval()
        runner(val_loader, 'valid')
        best_loss = runner.loss()

    torch.save(model.state_dict(), f'{save_dir}/latest.pt')
    shutil.copy(f'{save_dir}/latest.pt', f'{save_dir}/best.pt')

    try:
        for epoch in range(epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, config.epochs))
            print('-' * 10) 

            if hasattr(model, 'update') and callable(model.update):
                model.update(epoch + 1)

            model.train()
            runner(train_loader, 'train')

            if val_loader:
                model.eval()
                runner(val_loader, 'valid')
                metrics_dict = {k:v.item() for k,v in runner.latest['metrics'].items()}
                if use_nni:
                    nni.report_intermediate_result({'default':runner.loss().item(),**metrics_dict})

            torch.save(model.state_dict(), f'{save_dir}/latest.pt')
            if runner.loss() < best_loss:
                shutil.copy(f'{save_dir}/latest.pt', f'{save_dir}/best.pt')
                best_loss = runner.loss()
                print(f'Saved new best: {best_loss:.4}')

        if use_nni:
            metrics_dict = {k:v.item() for k,v in runner.latest['metrics'].items()}
            nni.report_final_result({'default':best_loss.item(),**metrics_dict})
    
    # Allow safe interruption of training loop
    except KeyboardInterrupt:
        print('\n\nExiting with honour\n')
        pass

    except Exception as e:
        print('\n\nDishonourable exit\n')
        raise e

    on_exit(config, runner, save_dir)

    writer.close()
    tee.flush()
    tee.close()
