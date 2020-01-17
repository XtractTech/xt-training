import os
import shutil
from importlib.util import spec_from_file_location, module_from_spec


def template(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    shutil.copy(os.path.join(dir_path, 'config_example.py'), args.save_dir)


def _import_config(path):
    """Import a config file given a file path."""
    config_spec = spec_from_file_location("config", path)
    config = module_from_spec(config_spec)
    config_spec.loader.exec_module(config)

    return config
