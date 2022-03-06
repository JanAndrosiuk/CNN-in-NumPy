import numpy as np
import opt_einsum as oe


def random_filters(no_filters, size=5, features=3):
    """
    Generate random filters for convolution function
    :param no_filters
    :param size: filter height and width (we assume here that height == width)
    :param features: number of input and filter features
        (first convolution takes rgb image as default, so it corresponds to 3 features)
    :return:
    """
    filters_matrix = np.random.randn(
        no_filters, size, size, features
    )

    return filters_matrix


def conv1(img_array, filters_matrix, use_oe):
    """
    First convolution
    :param img_array: input
    :param filters_matrix: generated filters
    :param use_oe: optimize np.einsum operation
    :return: convolved input
    """
    windowed = np.lib.stride_tricks.sliding_window_view(
        img_array,
        window_shape=(filters_matrix.shape[1], filters_matrix.shape[2]),
        axis=(1, 2)
    )
    if use_oe:
        return oe.contract('nabjkl,iklj->niab', windowed, filters_matrix)
    else:
        return np.einsum('nabjkl,iklj->niab', windowed, filters_matrix, optimize=True)


def conv2(conv_matrix, filters_matrix, use_oe):
    """
    Second convolution function
    Take output of the convolution as input and produce feature maps for every given filter
    SHAPE of conv_matrix: [previous_conv_no_filters x height x width]
    SHAPE of filters_matrix: [no_new_filters x filter_height x filter_width x previous_conv_no_filters]
    """

    # Split windows
    conv_matrix = np.lib.stride_tricks.sliding_window_view(
        conv_matrix,
        window_shape=(filters_matrix.shape[1], filters_matrix.shape[2]),
        axis=(2, 3)
    )

    # perform multiplication using einstein notation
    if use_oe:
        return oe.contract('albcjk,ijkl->aibc', conv_matrix, filters_matrix)
    else:
        return np.einsum('albcjk,ijkl->aibc', conv_matrix, filters_matrix, optimize=True)


def max_pool(matrix, pool):
    """
    Perform Max Pooling with
    This approach tries to split paddings +/- equally (more to right and bottom in some cases)
    # Larger pool size -> looping is faster
    # smaller pool size -> numpy operations are faster
    """

    n_filters = matrix.shape[1]

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

    split = padded.reshape(
        n_filters,
        padded.shape[2] // pool[0],
        pool[0],
        -1,
        pool[1]
    ).swapaxes(3, 4).reshape(matrix.shape[0], n_filters, -1, pool[0], pool[1])

    return np.max(
        np.max(
            split, 4
        ).swapaxes(2, 3), 2).reshape(
            matrix.shape[0],
            n_filters,
            padded.shape[2] // pool[0],
            padded.shape[3] // pool[1]
    )


def relu_fun(x):
    """
    Rectified Linear Unit activation function
    This function overwrites the input in-place
    :param x: input
    """
    x[x < 0] = 0
    return 1


def dense_layers(flat_input_shape, nodes):
    """
    Generate list of weight matrices
    :param flat_input_shape: flattened shape of convolutions and max-pooling result
        necessary to define height of first dense weight matrix
    :param nodes: number of units for each layer
    :return: list of dense weight matrices
    """

    # First element is of the size of the non-flattened pooled matrix
    weights_matrices = [
        np.random.normal(
            size=(flat_input_shape, nodes[0])
        )
    ]

    # Then
    for i in range(1, len(nodes)):
        weights_matrices.append(np.random.normal(size=(nodes[i - 1], nodes[i])))

    return weights_matrices


def softmax_fun(output_array):
    """
    Softmax activation function
    :param output_array: input matrix
    :return: input matrix filtered by Softmax function
    """
    return np.exp(output_array) / np.sum(np.exp(output_array), axis=1)


def normalize_features(input_matrix):
    """
    Normalize features (first dimension) of the input matrix
    :param input_matrix: expected shape: [number of features x height x width x other dimensions]
        yields stack of normalized features (output shape is the same as input)
    This function overwrites the input in-place
    """
    for i in range(input_matrix.shape[0]):
        input_matrix[i] = (input_matrix[i] - np.mean(input_matrix[i])) / np.std(input_matrix[i])
    return 1
