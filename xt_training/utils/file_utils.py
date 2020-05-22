import os
import shutil
from importlib.util import spec_from_file_location, module_from_spec


def template(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    shutil.copy(os.path.join(dir_path, 'config_example.py'), args.save_dir)
    if args.nni:
        nni_path = os.path.join(dir_path, 'nni')
        shutil.copy(os.path.join(nni_path, 'config_example.py'), args.save_dir)
        shutil.copy(os.path.join(nni_path, 'nni_config_example.yml'), args.save_dir)
        shutil.copy(os.path.join(nni_path, 'search_space_example.json'), args.save_dir)




def _import_config(path):
    """Import a config file given a file path."""
    config_spec = spec_from_file_location("config", path)
    config = module_from_spec(config_spec)
    config_spec.loader.exec_module(config)

    return config
