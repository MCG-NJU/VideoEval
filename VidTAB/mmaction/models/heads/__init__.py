# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseHead
from .gcn_head import GCNHead
from .i3d_head import I3DHead
from .dual_i3d_head import DualI3DHead
from .mvit_head import MViTHead
from .omni_head import OmniHead
from .slowfast_head import SlowFastHead
from .timesformer_head import TimeSformerHead
from .tpn_head import TPNHead
from .trn_head import TRNHead
from .tsm_head import TSMHead
from .tsn_audio_head import TSNAudioHead
from .tsn_head import TSNHead
from .x3d_head import X3DHead
from .pretrained_head import PretrainedHead
from .qa_head import AnswerHead

__all__ = [
    'BaseHead', 'GCNHead', 'I3DHead', 'MViTHead', 'OmniHead', 'SlowFastHead',
    'TPNHead', 'TRNHead', 'TSMHead', 'TSNAudioHead', 'TSNHead',
    'TimeSformerHead', 'X3DHead', 'DualI3DHead', 'AnswerHead'
]

