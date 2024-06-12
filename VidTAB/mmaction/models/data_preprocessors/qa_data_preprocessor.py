# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from mmengine.model import BaseDataPreprocessor, stack_batch

from mmaction.registry import MODELS
import time

@MODELS.register_module()
class QADataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for video quesion answer tasks.

    Args:
    data_list: Names of Parameter that need to be fed to the model

    """

    def __init__(self, data_list: Tuple[str], non_blocking: bool=False) -> None:
        super().__init__(non_blocking=non_blocking)
        self.data_list = data_list

    def forward(self,
                data: Union[dict, Tuple[dict]]) -> Union[dict, Tuple[dict]]:
        new_data = {}
        for k in data.keys():
            if k in self.data_list:
                new_data[k] = self.cast_data(data[k])

        return {'inputs': new_data}


