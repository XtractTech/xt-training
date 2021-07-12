import git
import mlflow
import os
import pytest

from pathlib import Path
from xt_training.utils.logging import _save_state, _save_config


def create_file(file_path):
    open(file_path, "a").close()


@pytest.fixture
def temp_repository_path(tmp_path):
    return tmp_path.joinpath("temp_repo")


@pytest.fixture
def current_config_path(tmp_path):

    tmp_path = tmp_path.joinpath("current_config")

    config_path = Path(tmp_path).joinpath("current_config")
    config_path.mkdir(parents=True)

    config_file_path = config_path.joinpath("current_config.py")
    create_file(config_file_path)

    return config_file_path


@pytest.fixture(name="repo")
def dummy_git_repo(temp_repository_path):

    repo_path = temp_repository_path
    repo = git.Repo.init(repo_path)

    file_name_temp = "text_file.txt"

    file_path_temp = repo_path.joinpath(file_name_temp)

    create_file(file_path_temp)
    repo.index.add([file_name_temp])
    repo.index.commit("dummy commit")

    file_path_temp.write_text("this text was not committed")

    create_file(repo_path.joinpath("an_untracked_file.py"))

    return repo


@pytest.fixture
def mlflow_tracking_path(tmp_path):
    output_path = f"{str(tmp_path)}/mlruns"
    mlflow.set_tracking_uri(f"file://{output_path}")
    mlflow.set_experiment("test_experiment")
    mlflow.start_run()
    yield Path(output_path)
    mlflow.end_run()


def test_save_state(
    repo, temp_repository_path, current_config_path, mlflow_tracking_path
):

    save_dir = temp_repository_path
    repo_path = repo.git.rev_parse("--show-toplevel")

    expected_commit_hash = repo.head.object.hexsha
    expected_untracked_file_name = "an_untracked_file.py"
    expected_diff_text = "this text was not committed"

    os.chdir(repo_path)
    _save_config(save_dir, str(current_config_path))
    _save_state(save_dir, mlflow_log=True)

    expected_output_file_path = Path(save_dir).joinpath("git.patch")
    assert (
        expected_output_file_path.exists()
    ), "git.patch file was not successfully created"

    output_string = expected_output_file_path.read_text()
    assert expected_commit_hash in output_string, "Commit hash was not as expected"
    assert (
        expected_untracked_file_name in output_string
    ), "Untracked file list is incorrect"
    assert expected_diff_text in output_string, "Did not record diff of file"

    run = mlflow.active_run()

    mlflow_git_logs_path = mlflow_tracking_path.joinpath(
        run.info.experiment_id, run.info.run_id, "artifacts", "git_logs"
    )

    assert mlflow_git_logs_path.joinpath("git.patch",).exists()

    assert mlflow_git_logs_path.joinpath(
        "untracked_files", "an_untracked_file.py",
    ).exists()

    assert mlflow_git_logs_path.joinpath("untracked_files", "config.py",).exists()

