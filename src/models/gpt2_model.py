import os
from typing import Any, Dict, Optional, List, Tuple
from encoders import GPT2Encoder

from .model import Model
from dpu_utils.utils import RichPath
import numpy as np
import random
import wandb

LoadedSamples = Dict[str, List[Dict[str, Any]]]
SampleId = Tuple[str, int]


class GPT2Model(Model):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        # config = GPT2Config()
        hypers = {}
        for label in ["code", "query"]:
            hypers.update({f'{label}_{key}': value
                           for key, value in GPT2Encoder.get_default_hyperparameters().items()})
        model_hypers = {
            'learning_rate': 5e-4,
            'code_use_subtokens': False,
            'code_mark_subtoken_end': False,
            'batch_size':2,
	    'max_epochs':2
        }
        hypers.update(super().get_default_hyperparameters())
        hypers.update(model_hypers)
        return hypers

    def __init__(self,
                 hyperparameters: Dict[str, Any],
                 run_name: str = None,
                 model_save_dir: Optional[str] = None,
                 log_save_dir: Optional[str] = None):

        super().__init__(
            hyperparameters,
            code_encoder_type=GPT2Encoder,
            query_encoder_type=GPT2Encoder,
            run_name=run_name,
            model_save_dir=model_save_dir,
            log_save_dir=log_save_dir)

