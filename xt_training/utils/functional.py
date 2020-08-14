import os
import shutil
import nni
from importlib.util import spec_from_file_location, module_from_spec

import torch
from torch.utils.tensorboard import SummaryWriter
from xt_training import Runner, metrics
from xt_training.utils import _import_config, Tee


def train_exit(config, runner, save_dir):
    test_loaders = getattr(config, 'test_loaders', None)

    # Final evaluation against test set(s)
    if test_loaders:
        print('\nTest')
        print('-' * 10)
        config.model.eval()
        for loader_name, loader in test_loaders.items():
            runner(loader, loader_name)


def train(
    save_dir,
    train_loader,
    model,
    optimizer,
    epochs,
    loss_fn,
    overwrite=True,
    val_loader=None,
    test_loaders=None,
    scheduler=None,
    eval_metrics={'eps': metrics.EPS()},
    on_exit=train_exit,
    use_nni=False
):
    if use_nni:
        save_dir = os.getenv("NNI_OUTPUT_DIR", save_dir)

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    if hasattr(loss_fn, 'weight') and loss_fn.weight is not None:
        loss_fn.weight = loss_fn.weight.to(device)
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

    runner.save_model(save_dir, True)

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

            if runner.loss() < best_loss:
                runner.save_model(save_dir, True)
                best_loss = runner.loss()
                print(f'Saved new best: {best_loss:.4}')
            else:
                runner.save_model(save_dir, False)

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


def test_exit(config, runner, save_dir):
    pass


def test(
    save_dir,
    config_path,
    checkpoint_path,
    model,
    val_loader=None,
    test_loaders=None,
    loss_fn=lambda *_: torch.tensor(0.),
    eval_metrics={'eps': metrics.EPS()},
    on_exit=test_exit
):
    # Initialize logging
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        try:
            if isinstance(config_path, str):
                shutil.copy(config_path, f'{save_dir}/config.py')
            else:
                shutil.copy(config_path['__file__'], f'{save_dir}/config.py')
        except shutil.SameFileError:
            pass
        tee = Tee(os.path.join(save_dir, "test.log"))
        results = {}

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
