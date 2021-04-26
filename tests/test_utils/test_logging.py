import git
import os
import pytest

from pathlib import Path
from xt_training.utils.logging import _save_state

def create_file(file_path):
    open(file_path, 'a').close()

@pytest.fixture
def current_config_path(tmp_path):

    config_path = Path(tmp_path).joinpath('current_config')
    config_path.mkdir(parents=True)

    config_file_path = config_path.joinpath('current_config.py')
    create_file(config_file_path)

    return config_file_path


@pytest.fixture(name='repo')
def dummy_git_repo(tmp_path):

    repo_path = tmp_path
    repo = git.Repo.init(repo_path)

    file_name_temp = 'text_file.txt'

    file_path_temp = repo_path.joinpath(file_name_temp)

    create_file(file_path_temp)
    repo.index.add([file_name_temp])
    repo.index.commit('dummy commit')

    Path(file_path_temp).write_text('this text was not committed')

    create_file(repo_path.joinpath('an_untracked_file.py'))

    return repo


def test_save_state(repo, tmp_path, current_config_path):

    save_dir = tmp_path
    repo_path = repo.git.rev_parse("--show-toplevel")
    
    expected_commit_hash = repo.head.object.hexsha
    expected_untracked_file_name = 'an_untracked_file.py'
    expected_diff_text = 'this text was not committed'
    
    os.chdir(repo_path)
    _save_state(save_dir, str(current_config_path))

    expected_output_file_path = Path(save_dir).joinpath('git.patch')
    assert expected_output_file_path.exists(), 'git.patch file was not successfully created'

    output_string = expected_output_file_path.read_text()
    assert expected_commit_hash in output_string, 'Commit hash was not as expected'
    assert expected_untracked_file_name in output_string, 'Untracked file list is incorrect'
    assert expected_diff_text in output_string, 'Did not record diff of file'

