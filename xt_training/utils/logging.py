import os
import sys
import shutil
import readline

import git

import __main__ as main


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


def _save_config(save_dir, config_path):
    # Save config file
    try:
        if isinstance(config_path, str):
            shutil.copy(config_path, os.path.join(save_dir, 'config.py'))
        else:
            shutil.copy(config_path['__file__'], os.path.join(save_dir, 'config.py'))
    except shutil.SameFileError:
        pass


PATCH_HEADER = '''
To replicate the git state used for this checkpoint, run the following:

    $ git checkout <commit hash shown below>
    $ git apply <path to this file>
'''

SESSION_HEADER = '''
"""The following code represents the most recent 200 lines of the python history leading up to calling
xt-training from an interactive python session. This is recorded to allow for reproduction of
experiments. However, this is only possible if 200 lines (or less) of python history are sufficient
to describe the state at runtime. For cleaner state logging, try using an IPython console instead 
(ipython or jupyter)."""

'''


def _save_state(save_dir):
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

    # Save session history (for interactive sessions only)
    if hasattr(sys, 'ps1'):
        # We are running in an interactive session
        try:
            if hasattr(main, 'In'):
                # This is an IPython session (includes Jupyter)
                with open(os.path.join(save_dir, 'session-ipython.py'), 'w') as f:
                    f.write('\n'.join(main.In))
            else:
                # Otherwise assume we are in the built-in python console
                with open(os.path.join(save_dir, 'session.py'), 'w') as f:
                    f.write(SESSION_HEADER)
                    history_len = readline.get_current_history_length()
                    # No way to limit to this session for python console
                    # Save only the most recent 200 lines
                    for i in range(max(history_len-200, 0), history_len):
                        f.write(readline.get_history_item(i + 1) + '\n')                    
        except:
            pass
