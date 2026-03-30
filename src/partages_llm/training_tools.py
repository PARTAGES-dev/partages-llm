import json
from pathlib import Path
from typing import Any, Dict, List, Union
from itertools import product

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
from torch.profiler import profile


class ProfilerCallback(TrainerCallback):

    def __init__(self, profiler: profile):
        self.profiler = profiler

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        self.profiler.step()


def unroll_config(
    config: Union[Path, Dict[str, Any]],
    return_list: bool = False
) -> Union[List[Dict[str, Any]], product]:
    """
    Helper function for preprocessing a hyperparameter grid configuration into an iterable over the unique
    value combinations to be tested. To be used when a grid search is to be distributed across multiple
    GPU jobs.
    """
    if isinstance(config, Path):
        with config.open() as f:
            config = json.load(f)
    config_processed = {k: [True, False] if v is None else v for k, v in config.items()}
    unrolled_config = product(*config_processed.values())
    if return_list:
        return [dict(zip(config_processed, value_combination)) for value_combination in unrolled_config]
    return unrolled_config

