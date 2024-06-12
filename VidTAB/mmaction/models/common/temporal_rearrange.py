import math


def temporal_rearrange_NCHW(x, rearrange_type, origin_shape, window_size, inverse=False):
    """
    input: N*T C H W
    output: N C th*H tw*W
    inverse则相反
    """
    N, T, C, H, W = origin_shape
    t_h = int(math.sqrt(T))
    t_w = T // t_h
    # 目前先定死用9, 16这样能整数开方的输入吧，方便
    assert t_w == t_h, f"t_w({t_w}) != t_h({t_h})"
    if inverse:
        if rearrange_type == 'local':  # way4 复原
            x = x.view(N, C, H//window_size[0], t_h, window_size[0], W//window_size[1], t_w,
                       window_size[1]).permute(0, 3, 6, 1, 2, 4, 5, 7).contiguous().view(N*T, C, H, W)
        elif rearrange_type == 'global':  # way1 复原
            x = x.view(N, C, t_h, H, t_w, W).permute(0, 2, 4, 1, 3, 5).contiguous().view(N*T, C, H, W)
        else:
            raise NotImplementedError(
                f"Not support rearrange_type: {rearrange_type}!")
    else:
        if rearrange_type == 'local':  # way4 stack
            x = x.view(N, t_h, t_w, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1]).permute(
                0, 3, 4, 1, 5, 6, 2, 7).contiguous().view(N, C, t_h*H, t_w*W)
        elif rearrange_type == 'global':  # way1 stack
            x = x.view(N, t_h, t_w, C, H, W).permute(
                0, 3, 1, 4, 2, 5).contiguous().view(N, C, t_h*H, t_w*W)
        else:
            raise NotImplementedError(
                f"Not support rearrange_type: {rearrange_type}!")

    return x

def temporal_rearrange_NCTHW(x, rearrange_type, origin_shape, window_size, inverse=False):
    """
    input: N C T H W
    output: N C th*H tw*W
    inverse则相反
    """
    N, C, T, H, W = origin_shape
    t_h = int(math.sqrt(T))
    t_w = T // t_h
    # 目前先定死用9, 16这样能整数开方的输入吧，方便
    assert t_w == t_h, f"t_w({t_w}) != t_h({t_h})"
    if inverse:
        if rearrange_type == 'local':  # way4 复原
            x = x.view(N, C, H//window_size[0], t_h, window_size[0], W//window_size[1], t_w, window_size[1]).permute(
                0, 1, 3, 6, 2, 4, 5, 7).contiguous().view(N, C, T, H, W)
        elif rearrange_type == 'global':  # way1 复原
            x = x.view(N, C, t_h, H, t_w, W).permute(
                0, 1, 2, 4, 3, 5).contiguous().view(N, C, t_h*t_w, H, W)
        else:
            raise NotImplementedError(
                f"Not support rearrange_type: {rearrange_type}!")
    else:
        if rearrange_type == 'local':  # way4 stack
            x = x.view(N, C, t_h, t_w, H//window_size[0], window_size[0], W//window_size[1], window_size[1]).permute(
                0, 1, 4, 2, 5, 6, 3, 7).contiguous().view(N, C, t_h*H, t_w*W)
        elif rearrange_type == 'global':  # way1 stack
            x = x.view(N, C, t_h, t_w, H, W).permute(
                0, 1, 2, 4, 3, 5).contiguous().view(N, C, t_h*H, t_w*W)
        else:
            raise NotImplementedError(
                f"Not support rearrange_type: {rearrange_type}!")

    return x