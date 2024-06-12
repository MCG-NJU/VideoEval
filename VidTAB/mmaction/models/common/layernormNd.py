import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MMENGINE_MODELS


@MMENGINE_MODELS.register_module('LN2d')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape,
            self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()

@MMENGINE_MODELS.register_module('LN3d')
class LayerNorm3d(nn.LayerNorm):
    """LayerNorm on channels for 3d video.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 5, 'LayerNorm3d only supports inputs with shape ' \
            f'(N, C, T, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 4, 1).contiguous(), self.normalized_shape,
            self.weight, self.bias, self.eps).permute(0, 4, 1, 2, 3).contiguous()