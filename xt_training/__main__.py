import argparse

from .utils import train, template, test


def parse_args():
    parser = argparse.ArgumentParser(
        description='A tool for training/evaluating pytorch deep learning models'
    )
    subparsers = parser.add_subparsers(help='Subcommand help')

    parser_train = subparsers.add_parser(
        'train',
        help='Train a model given a config script.',
        description='Train a model given a config script.'
    )
    parser_train.add_argument(
        'config_path',
        type=str,
        help='Path to fully specified config.'
    )
    parser_train.add_argument(
        'save_dir',
        type=str,
        help='Path to save model checkpoints, tensorboard runs, logs, and config files.'
    )
    parser_train.add_argument(
        '--overwrite', '-o',
        action='store_true',
        help='Switch to allow overwriting of existing path if necessary.'
    )
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser(
        'test',
        help='Test a model given a config script.',
        description='Test a model given a config script.'
    )
    parser_test.add_argument(
        'config_path',
        type=str,
        help='Path to fully specified config (if a file) or checkpoint directory (if a directory).'
    )
    parser_test.add_argument(
        'checkpoint_path',
        type=str,
        help='Path to compatible model checkpoint.',
        default=None,
        nargs='?'
    )
    parser_test.add_argument(
        '--save_dir',
        type=str,
        help='Path to save logs, config files, and (optionally) predictions.',
        default=None
    )
    parser_test.set_defaults(func=test)

    parser_template = subparsers.add_parser(
        'template',
        help='Generate a config script template.',
        description='Generate a config script template.'
    )
    parser_template.add_argument(
        'save_dir',
        type=str,
        help='Path to save config templates.'
    )
    parser_template.add_argument(
        '--nni',
        action='store_true',
        help='Generate nni template files.'
    )
    parser_template.set_defaults(func=template)
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    args.func(args)
