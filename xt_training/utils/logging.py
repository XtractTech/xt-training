import sys
import shutil

import git


class Tee(object):
    """Object that replicates the functionality of the `tee` shell command."""

    def __init__(self, logfile):
        """Constructor for Tee class.
        
        Arguments:
            logfile {str} -- Path to destination file.
        """
        self.file = open(logfile, "w")
        self.stdout = sys.stdout
        sys.stdout = self
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.close()


PATCH_HEADER = '''
To replicate the git state used for this checkpoint, run the following:

    $ git checkout <commit hash shown below>
    $ git apply <path to this file>
'''


def _save_state(save_dir, config_path):
    # Save config file
    try:
        if isinstance(config_path, str):
            shutil.copy(config_path, os.path.join(save_dir, 'config.py'))
        else:
            shutil.copy(config_path['__file__'], os.path.join(save_dir, 'config.py'))
    except shutil.SameFileError:
        pass

    # If we are in a git repo, save git state file
    try:
        repo = git.Repo('.')
        commit = repo.head.object.hexsha
        untracked = repo.untracked_files
        diff = repo.git.diff()

        with open(os.path.join(save_dir, 'git.patch'), 'w') as f:
            f.write(PATCH_HEADER)
            f.write(f'\n\nCommit: {commit}')
            f.write('\n\nUntracked files:\n')
            f.write('\n'.join(untracked))
            f.write('\n\n')
            f.write(diff)
    # Silently skip if no git repo is found
    except:
        pass