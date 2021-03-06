import numpy as np
import opt_einsum as oe


def relu_prime(z):
    """
    Derivative of the Rectified Linear Unit activation function
    :param z: Pre-activation result
    :return: derived input
    """
    return (z > 0).astype(z.dtype)


def block_4d(arr, window):
    """
    Used for max-pooling derivative
    Explanation is included in README references
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
    Used for max-pooling derivative
    Explanation is included in README references
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
    Derivative of MaxPool function
    The explanation of the mechanism is linked in README file -> References
    Makes use of block shaped matrices, which are also included in references
    :param matrix: input
    :param pool: pool size - usually (0,0); handles non-square shapes
    :return: max pool derivative matrix
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

    split_max_bool = (split == np.amax(split, axis=(3, 4), keepdims=1)).astype(float)
    return unblock_5d(split_max_bool, padded.shape[-2:])


def c2fprime(conv_matrix, filters_matrix, use_oe):
    """
    Second convolution derivative over filter matrix
    For now the function works only with 1 image at once
    later it should be generalized for multiple images (einsum function in return)
        e.g. with np.apply_along_axis() or np.apply_over_axes()
    # SHAPE of conv_matrix: [previous_conv_no_filters x height x width]
    # SHAPE of filters_matrix: [no_new_filters x filter_height x filter_width x previous_conv_no_filters]
    :param use_oe: optimize np.einsum operation
    """
    # print(conv_matrix.shape, filters_matrix.shape)

    conv_matrix = np.lib.stride_tricks.sliding_window_view(
        conv_matrix,
        window_shape=(filters_matrix.shape[2], filters_matrix.shape[3]),  # filter height and width
        axis=(2, 3)  # conv_matrix height and width
    )

    if use_oe:
        return oe.contract('abcjk,ijk->ibca', conv_matrix[0], filters_matrix[0])
    else:
        return np.einsum('abcjk,ijk->ibca', conv_matrix[0], filters_matrix[0], optimize=True)


def c2xprime(conv_matrix, filters_matrix, use_oe):
    """
    Second convolution derivative over convolution input
    conv_matrix: c2 filters matrix with 0 padding on height and height
        (value of those paddings = filters_matrix width and height - 1)
        # conv_matrix dim before padding = [n_filters x width x height x previous convolution n_filters]
    filters_matrix: derivation of the error over c2 output = delta
        # dim depends - in this case: [(1 x 64 x 24 x 24)]
    :param use_oe: optimize np.einsum operation
    """

    # horizontal and vertical paddings:
    pad_h, pad_v = filters_matrix.shape[-2:]

    conv_matrix = np.pad(conv_matrix, (
        (0, 0),  # no padding for n_filters dimension
        (pad_h - 1, pad_h - 1),
        (pad_v - 1, pad_v - 1),
        (0, 0)  # no padding for number of features
    ), 'constant')

    conv_matrix = np.lib.stride_tricks.sliding_window_view(
        conv_matrix,
        window_shape=(filters_matrix.shape[2], filters_matrix.shape[3]),  # filter height and width
        axis=(1, 2)  # conv_matrix height and width
    )

    if use_oe:
        return oe.contract('iyzxjk,ijk->xyz', conv_matrix, filters_matrix[0])
    else:
        return np.einsum('iyzxjk,ijk->xyz', conv_matrix, filters_matrix[0], optimize=True)


def c1fprime(conv_matrix, filters_matrix, use_oe):
    """
    Returns first convolution derivative over 1st filter matrix (w0)
    conv_matrix - input image (x_train in CNN class (model.py))
        e.g. [1, 28, 28, 1]
    filters_matrix - delta
        e.g. dim [1 x 32 x 26 x 26]
    1st dims of those matrices stand for batch size and are not supported yet
    :param use_oe: optimize np.einsum operation
    """

    conv_matrix = np.lib.stride_tricks.sliding_window_view(
        conv_matrix,
        window_shape=(filters_matrix.shape[2], filters_matrix.shape[3]),  # filter height and width
        axis=(1, 2)  # conv_matrix height and width
    )

    if use_oe:
        return oe.contract('xyzbc,nbc->nxyz', conv_matrix[0], filters_matrix[0])
    else:
        return np.einsum('xyzbc,nbc->nxyz', conv_matrix[0], filters_matrix[0], optimize=True)
