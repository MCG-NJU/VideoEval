# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmengine.model import BaseModel, merge_dict
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)

@MODELS.register_module()
class BaseQuestionAnswer(BaseModel, metaclass=ABCMeta):
    """Base class for Video QuestionAnswer.

    Args:
        backbone (Union[ConfigDict, dict]): Backbone modules to
            extract feature.
        head (Union[ConfigDict, dict]): QA head to produce output.
        neck (Union[ConfigDict, dict]): Neck for feature fusion.
        train_cfg (Union[ConfigDict, dict], optional): Config for training.
            Defaults to None.
        test_cfg (Union[ConfigDict, dict], optional): Config for testing.
            Defaults to None.
        data_preprocessor (Union[ConfigDict, dict], optional): The pre-process
           config of :class:`QADataPreprocessor`. 
    """

    def __init__(self,
                 backbone: ConfigType,
                 head: ConfigType,
                 neck: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None) -> None:
        if data_preprocessor is None:
            # This preprocessor will only stack batch data samples.
            data_preprocessor = dict(type='QADataPreprocessor')

        super(BaseRecognizer,
              self).__init__(data_preprocessor=data_preprocessor)

        # Record the source of the backbone.
        self.backbone_from = 'mmaction2'

        if backbone['type'].startswith('mmcls.'):
            try:
                # Register all mmcls models.
                import mmcls.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this backbone.')
            self.backbone = MODELS.build(backbone)
            self.backbone_from = 'mmcls'
        elif backbone['type'].startswith('torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this '
                                  'backbone.')
            backbone_type = backbone.pop('type')[12:]
            self.backbone = torchvision.models.__dict__[backbone_type](
                **backbone)
            # disable the classifier
            self.backbone.classifier = nn.Identity()
            self.backbone.fc = nn.Identity()
            self.backbone_from = 'torchvision'
        elif backbone['type'].startswith('timm.'):
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm to use this '
                                  'backbone.')
            backbone_type = backbone.pop('type')[5:]
            # disable the classifier
            backbone['num_classes'] = 0
            self.backbone = timm.create_model(backbone_type, **backbone)
            self.backbone_from = 'timm'
        elif backbone['type'] == 'feature':
            self.backbone = nn.Identity()
        else:
            self.backbone = MODELS.build(backbone)

        self.neck = MODELS.build(neck)
        self.head = MODELS.build(head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self) -> None:
        """Initialize the model network weights."""
        super().init_weights()
        if self.backbone_from in ['torchvision', 'timm']:
            warnings.warn('We do not initialize weights for backbones in '
                          f'{self.backbone_from}, since the weights for '
                          f'backbones in {self.backbone_from} are initialized '
                          'in their __init__ functions.')


    def forward(self,
                inputs: dict,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes:

        - ``tensor``: Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - ``predict``: Forward and return the predictions, which are fully
        processed to a list of :obj:`ActionDataSample`.
        - ``loss``: Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (dict): The input dict.
        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of ``ActionDataSample``.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode in ['tensor', 'predict', 'loss']:
            return self.neck(inputs=inputs, backbone=self.backbone, head=self.head, mode=mode)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
