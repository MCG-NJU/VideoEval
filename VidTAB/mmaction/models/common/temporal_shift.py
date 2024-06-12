
import torch
from torch import nn


def temporal_difference_NCTHW(x, shift_div=3, inplace=False):
    """
    和tsm还有点不一样，这个不用零填充，我这个实现方式是inplace操作
    """
    # [N, C, T, H, W]
    c = x.size()[1]

    # NOTE: can't use 5 dimensional array on PPL2D backend for caffe
    # x = x.view(-1, c, t, h * w)

    # get shift fold
    fold = c // shift_div

    if inplace:
        x_shift = x[:, :2*fold, :, :, :].clone()
    else:
        x_shift = x[:, :2*fold, :, :, :]
        x = x.clone()

    x[:, :fold, :-1, :, :] = x_shift[:, :fold, :-1, :, :] - \
        x_shift[:, :fold, 1:, :, :]  # diff right
    # x[:, :fold, -1 ,:, :] = 0
    x[:, fold:2*fold, 1:, :, :] = x_shift[:, fold:2*fold, 1:, :, :] - \
        x_shift[:, fold:2*fold, :-1, :, :]  # diff left
    # x[:, fold:2*fold, 0 ,:, :] = 0

    # [N, C, T, H, W]
    # restore the original dimension
    return x


def temporal_difference_NCHW(x, num_segments, shift_div=3, inplace=False):
    """
    和tsm还有点不一样，这个不用零填充，我这个实现方式是inplace操作
    """
    # [N*T, C, H, W]
    NT, C, H, W = x.shape
    N, T = NT // num_segments, num_segments

    # NOTE: can't use 5 dimensional array on PPL2D backend for caffe
    x = x.view(N, T, C, H * W)

    # get shift fold
    fold = C // shift_div

    if inplace:
        x_shift = x[:, :, :2*fold, :].clone()
    else:
        x_shift = x[:, :, :2*fold, :]
        x = x.clone()

    x[:, :-1, :fold, :] = x_shift[:, :-1, :fold, :] - \
        x_shift[:, 1:, :fold, :]  # diff right
    # x[:, -1 , :fold, :] = 0
    x[:, 1:, fold:2*fold, :] = x_shift[:, 1:, fold:2*fold, :] - \
        x_shift[:, :-1, fold:2*fold, :]  # diff left
    # x[:, 0 , fold:2*fold, :] = 0

    # [N, C, T, H, W]
    # restore the original dimension
    return x.view(N*T, C, H, W)


def tsm_shift_NCTHW(x, shift_div=3):
    """Perform temporal shift operation on the feature.

    Args:
        x (torch.Tensor): The input feature to be shifted.
        num_segments (int): Number of frame segments.
        shift_div (int): Number of divisions for shift. Default: 3.

    Returns:
        torch.Tensor: The shifted feature.
    """
    # [N, C, T, H, W]
    n, c, t, h, w = x.size()

    # NOTE: can't use 5 dimensional array on PPL2D backend for caffe
    x = x.view(-1, c, t, h * w)

    # get shift fold
    fold = c // shift_div

    # split c channel into three parts:
    # left_split, mid_split, right_split
    left_split = x[:, :fold, :, :]
    mid_split = x[:, fold:2 * fold, :, :]
    right_split = x[:, 2 * fold:, :, :]

    # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
    # because array on caffe inference must be got by computing

    # shift left on num_segments channel in `left_split`
    zeros = left_split - left_split
    blank = zeros[:, :, :1, :]
    left_split = left_split[:, :, 1:, :]
    left_split = torch.cat((left_split, blank), 2)

    # shift right on num_segments channel in `mid_split`
    zeros = mid_split - mid_split
    blank = zeros[:, :, :1, :]
    mid_split = mid_split[:, :, :-1, :]
    mid_split = torch.cat((blank, mid_split), 2)

    # right_split: no shift

    # concatenate
    out = torch.cat((left_split, mid_split, right_split), 2)

    # [N, C, T, H, W]
    # restore the original dimension
    return out.view(n, c, t, h, w)


def tsm_shift_NCHW(x, num_segments, shift_div=3):
    """Perform temporal shift operation on the feature.

    Args:
        x (torch.Tensor): The input feature to be shifted.
        num_segments (int): Number of frame segments.
        shift_div (int): Number of divisions for shift. Default: 3.

    Returns:
        torch.Tensor: The shifted feature.
    """
    # [N, C, H, W]
    n, c, h, w = x.size()

    # [N // num_segments, num_segments, C, H*W]
    # can't use 5 dimensional array on PPL2D backend for caffe
    x = x.view(-1, num_segments, c, h * w)

    # get shift fold
    fold = c // shift_div

    # split c channel into three parts:
    # left_split, mid_split, right_split
    left_split = x[:, :, :fold, :]
    mid_split = x[:, :, fold:2 * fold, :]
    right_split = x[:, :, 2 * fold:, :]

    # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
    # because array on caffe inference must be got by computing

    # shift left on num_segments channel in `left_split`
    zeros = left_split - left_split
    blank = zeros[:, :1, :, :]
    left_split = left_split[:, 1:, :, :]
    left_split = torch.cat((left_split, blank), 1)

    # shift right on num_segments channel in `mid_split`
    zeros = mid_split - mid_split
    blank = zeros[:, :1, :, :]
    mid_split = mid_split[:, :-1, :, :]
    mid_split = torch.cat((blank, mid_split), 1)

    # right_split: no shift

    # concatenate
    out = torch.cat((left_split, mid_split, right_split), 2)

    # [N, C, H, W]
    # restore the original dimension
    return out.view(n, c, h, w)


def tcs_shift_NCHW(x, num_segments, shift_div=3, inplace=False):
    """
    和tsm还有点不一样，这个不用零填充
    """
    # [N*T, C, H, W]
    NT, C, H, W = x.shape
    N, T = NT // num_segments, num_segments

    # NOTE: can't use 5 dimensional array on PPL2D backend for caffe
    x = x.view(N, T, C, H * W)

    # get shift fold
    fold = C // shift_div

    if inplace:
        x_shift = x[:, :, :2*fold, :].clone()
    else:
        x_shift = x[:, :, :2*fold, :]
        x = x.clone()

    x[:, :-1, :fold, :] = x_shift[:, 1:, :fold, :]  # shift left
    # x[:, -1 , :fold, :] = 0
    x[:, 1:, fold:2*fold, :] = x_shift[:, :-1, fold:2*fold, :]  # shift right
    # x[:, 0 , fold:2*fold, :] = 0

    # [N, C, T, H, W]
    # restore the original dimension
    return x.view(N*T, C, H, W)


def tcs_shift_NCTHW(x, shift_div=3, inplace=False):
    """
    和tsm还有点不一样，这个不用零填充
    """
    # [N, C, T, H, W]
    C = x.size()[1]

    # NOTE: can't use 5 dimensional array on PPL2D backend for caffe
    # x = x.view(-1, c, t, h * w)

    # get shift fold
    fold = C // shift_div
    if inplace:
        x_shift = x[:, :2*fold, :, :, :].clone()
    else:
        x_shift = x[:, :2*fold, :, :, :]
        x = x.clone()

    x[:, :fold, :-1, :, :] = x_shift[:, :fold, 1:, :, :]  # shift left
    # x[:, :fold, -1 ,:, :] = 0
    x[:, fold:2*fold, 1:, :, :] = x_shift[:,
                                          fold:2*fold, :-1, :, :]  # shift right
    # x[:, fold:2*fold, 0 ,:, :] = 0

    # [N, C, T, H, W]
    # restore the original dimension
    return x


def tps_shift_NCHW(x, num_segments, invert=False, ratio=1, stride=1, inplace=False):
    '''我这个实现方式是inplace操作'''
    # [N*T, C, H, W]
    NT, C, H, W = x.shape
    N, T = NT // num_segments, num_segments

    # NOTE: can't use 5 dimensional array on PPL2D backend for caffe
    x = x.view(N, T, C, H, W)

    fold = int(C * ratio)

    if inplace:
        x_shift = x[:, :, :fold, :].clone()
    else:
        x_shift = x[:, :, :fold, :]
        x = x.clone()

    if invert:
        multiplier = -1
    else:
        multiplier = 1

    # Pattern C in TPS
    x[:, :, :fold, 0::3, 0::3] = torch.roll(
        x_shift[:, :, :, 0::3, 0::3], shifts=-4*multiplier*stride, dims=1)
    x[:, :, :fold, 0::3, 1::3] = torch.roll(
        x_shift[:, :, :, 0::3, 1::3], shifts=multiplier*stride, dims=1)
    x[:, :, :fold, 1::3, 0::3] = torch.roll(
        x_shift[:, :, :, 1::3, 0::3], shifts=-multiplier*stride, dims=1)
    x[:, :, :fold, 0::3, 2::3] = torch.roll(
        x_shift[:, :, :, 0::3, 2::3], shifts=2*multiplier*stride, dims=1)
    x[:, :, :fold, 2::3, 0::3] = torch.roll(
        x_shift[:, :, :, 2::3, 0::3], shifts=-2*multiplier*stride, dims=1)
    x[:, :, :fold, 1::3, 2::3] = torch.roll(
        x_shift[:, :, :, 1::3, 2::3], shifts=3*multiplier*stride, dims=1)
    x[:, :, :fold, 2::3, 1::3] = torch.roll(
        x_shift[:, :, :, 2::3, 1::3], shifts=-3*multiplier*stride, dims=1)
    x[:, :, :fold, 2::3, 2::3] = torch.roll(
        x_shift[:, :, :, 2::3, 2::3], shifts=4*multiplier*stride, dims=1)

    return x.view(N*T, C, H, W)


def tps_shift_NCTHW(x, invert=False, shift_div=2, stride=1, inplace=False):
    '''我这个实现方式是inplace操作'''
    C = x.size()[1]
    fold = C // shift_div

    if inplace:
        x_shift = x[:, :fold, :, :, :].clone()
    else:
        x_shift = x[:, :fold, :, :, :]
        x = x.clone()

    if invert:
        multiplier = -1
    else:
        multiplier = 1

    # Pattern C in TPS
    x[:, :fold, :, 0::3, 0::3] = torch.roll(
        x_shift[:, :, :, 0::3, 0::3], shifts=-4*multiplier*stride, dims=2)
    x[:, :fold, :, 0::3, 1::3] = torch.roll(
        x_shift[:, :, :, 0::3, 1::3], shifts=multiplier*stride, dims=2)
    x[:, :fold, :, 1::3, 0::3] = torch.roll(
        x_shift[:, :, :, 1::3, 0::3], shifts=-multiplier*stride, dims=2)
    x[:, :fold, :, 0::3, 2::3] = torch.roll(
        x_shift[:, :, :, 0::3, 2::3], shifts=2*multiplier*stride, dims=2)
    x[:, :fold, :, 2::3, 0::3] = torch.roll(
        x_shift[:, :, :, 2::3, 0::3], shifts=-2*multiplier*stride, dims=2)
    x[:, :fold, :, 1::3, 2::3] = torch.roll(
        x_shift[:, :, :, 1::3, 2::3], shifts=3*multiplier*stride, dims=2)
    x[:, :fold, :, 2::3, 1::3] = torch.roll(
        x_shift[:, :, :, 2::3, 1::3], shifts=-3*multiplier*stride, dims=2)
    x[:, :fold, :, 2::3, 2::3] = torch.roll(
        x_shift[:, :, :, 2::3, 2::3], shifts=4*multiplier*stride, dims=2)

    return x
