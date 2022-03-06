import numpy as np


def relu_prime(z):
    return (z > 0).astype(z.dtype)


def block_4d(arr, window):
    """
    # https://stackoverflow.com/a/16873755
    4d -> 5d
    window -> (window_height, window_width)
    """
    return arr.reshape(
        arr.shape[1],
        arr.shape[2] // window[0],
        window[0],
        -1,
        window[1]
    ).swapaxes(3, 4).reshape(*arr.shape[:2], -1, *window)


def unblock_5d(arr, hw):
    """
    # https://stackoverflow.com/a/16873755
    5d -> 4d
    hw := (height, width)
    """
    return arr.reshape(
        arr.shape[1],
        hw[0] // arr.shape[3],
        -1,
        arr.shape[3],
        arr.shape[4]
    ).swapaxes(3, 4).reshape(*arr.shape[:2], *hw)


def max_pool_prime(matrix, pool):
    """
    Derivative over MaxPool function
    """

    # Residual of height and width division over respective pool dimensions
    res_h = matrix[0, 0].shape[0] % pool[0]
    res_v = matrix[0, 0].shape[1] % pool[1]

    padded = np.pad(matrix, (
        (0, 0),  # no padding for number_of_images
        (0, 0),  # no padding for filter dimension
        (
            res_h // 2,
            res_h // 2 + res_h % 2
        ),
        (
            res_v // 2,
            res_v // 2 + res_v % 2
        )
    ), 'constant')

    split = block_4d(padded, pool)

    # https://stackoverflow.com/a/42397281
    split_max_bool = (split == np.amax(split, axis=(3, 4), keepdims=1)).astype(float)
    return unblock_5d(split_max_bool, padded.shape[-2:])


def c2fprime(conv_matrix, filters_matrix):
    """
    Convolution derivative over filter matrix
    For now the function works only with 1 image at once
    later it should be generalized for multiple images (einsum function in return)
        e.g. with np.apply_along_axis() or np.apply_over_axes()
    # SHAPE of conv_matrix: [previous_conv_no_filters x height x width]
    # SHAPE of filters_matrix: [no_new_filters x filter_height x filter_width x previous_conv_no_filters]
    """
    # print(conv_matrix.shape, filters_matrix.shape)

    windowed = np.lib.stride_tricks.sliding_window_view(
        conv_matrix,
        window_shape=(filters_matrix.shape[2], filters_matrix.shape[3]),  # filter height and width
        axis=(2, 3)  # conv_matrix height and width
    )

    return np.einsum('abcjk,ijk->ibca', windowed[0], filters_matrix[0], optimize=True)


def c2xprime(conv_matrix, filters_matrix):
    """
    Convolution derivative over the input of this convolution
    conv_matrix: c2 filters matrix with 0 padding on height and height
        (value of those paddings = filters_matrix width and height - 1)
        # conv_matrix dim before padding = [n_filters x width x height x previous convolution n_filters]
    filters_matrix: derivation of the error over c2 output = delta
        # dim depends - in this case: [(1 x 64 x 24 x 24)]
    """

    # horizontal and vertical paddings:
    pad_h, pad_v = filters_matrix.shape[-2:]

    padded = np.pad(conv_matrix, (
        (0, 0),  # no padding for n_filters dimension
        (pad_h - 1, pad_h - 1),
        (pad_v - 1, pad_v - 1),
        (0, 0)  # no padding for number of features
    ), 'constant')

    windowed = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=(filters_matrix.shape[2], filters_matrix.shape[3]),  # filter height and width
        axis=(1, 2)  # conv_matrix height and width
    )

    return np.einsum('iyzxjk,ijk->xyz', windowed, filters_matrix[0], optimize=True)


def c1fprime(conv_matrix, filters_matrix):
    """
    Convolution derivative over 1st filter matrix (w0)
    conv_matrix - input image
        e.g. [1, 28, 28, 1]
    filters_matrix - delta
        e.g. dim [1 x 32 x 26 x 26]
    1st dims of those matrices stand for batch size and are not supported yet
    """

    windowed = np.lib.stride_tricks.sliding_window_view(
        conv_matrix,
        window_shape=(filters_matrix.shape[2], filters_matrix.shape[3]),  # filter height and width
        axis=(1, 2)  # conv_matrix height and width
    )

    return np.einsum('xyzbc,nbc->nxyz', windowed[0], filters_matrix[0], optimize=True)
