import datetime
import os
import tempfile
from typing import Callable

import numpy as np
from ray.rllib.algorithms import ppo
from ray.tune.logger import TBXLogger
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import UnifiedLogger


def get_current_time() -> str:
    """
    Generate current time as string.
    Returns:
        str: current time
    """
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    return curr_time


def make_logging_folder(root_dir: str, env_id: str, is_test: bool) -> str:
    """
    Make a folder for logging.
    Args:
        root_dir (str): parent directory
        env_id (str): env id name
        is_test (bool): whether to test the model

    Returns:
        str: maked logging folder name
    """
    task = "Train" if not is_test else "Test"
    logdir_prefix = "[{}]{}_{}_".format(task, get_current_time(), env_id)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=root_dir)
    return logdir


def make_initial_hidden_state(lstm_cell_size: int) -> list:
    """
    Make initial hidden state for testing lstm-based policy network.
    Args:
        lstm_cell_size (int): lstm cell size

    Returns:
        list: hidden state
    """
    hidden_state = [np.zeros(lstm_cell_size), np.zeros(lstm_cell_size)]
    return hidden_state


def save_model(trainer: ppo, path_to_save: str):
    """
    Save trained model.
    Args:
        trainer (Trainer): rllib trainer
    """
    trainer.save(path_to_save)


def load_model(trainer: ppo, path_to_load: str):
    """
    Load trained model.
    Args:
        trainer (Trainer): rllib trainer
        path_to_load (str): path to load
    """
    trainer.restore(path_to_load)


def get_logger_creator(logdir: str) -> Callable:
    """
    Get default logger creator for logging in rllib.
    Args:
        logdir (str): logging directory path

    Returns:
        Callable: logger creator
    """

    def logger_creator(config):
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


class CustomLogCallback(DefaultCallbacks):
    """
    LogCallback based on Rllib callbacks for add 'success' custom metric.
    """

    def on_episode_end(self, *, episode, **kwargs):
        """On episode end, add success custom metric."""
        info = episode.last_info_for()
        episode.custom_metrics["success"] = int(info["success"])