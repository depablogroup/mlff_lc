import dataclasses
import enum

import numpy as np
import pandas as pd

@dataclasses.dataclass
class ClusterInfo:
    slurm_template: str
    ntasks_per_node: int = 8


@dataclasses.dataclass
class RunInfo:
    class RunStatus(enum.Enum):
        NOT_STARTED = 0
        ONGOING = 1
        FAILED = 2
        NEED_RERUN = 3
        FINISHED = 4
    run_status: RunStatus = RunStatus.NOT_STARTED
    model_record: pd.DataFrame = None
    best_val_loss: float = np.inf

