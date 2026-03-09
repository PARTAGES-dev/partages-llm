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
