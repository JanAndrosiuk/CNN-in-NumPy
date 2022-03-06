import numpy as np


def random_filters(no_filters, size=5, features=3):
    filters_matrix = np.random.randn(
        no_filters, size, size, features
    )

    return filters_matrix


def conv1(img_array, filters_matrix):
    windowed = np.lib.stride_tricks.sliding_window_view(
        img_array,
        window_shape=(filters_matrix.shape[1], filters_matrix.shape[2]),
        axis=(1, 2)
    )

    return np.einsum('nabjkl,iklj->niab', windowed, filters_matrix, optimize=True)


def conv2(conv_matrix, filters_matrix):
    """
    Take output of the convolution as input and produce feature maps for every given filter
    SHAPE of conv_matrix: [previous_conv_no_filters x height x width]
    SHAPE of filters_matrix: [no_new_filters x filter_height x filter_width x previous_conv_no_filters]
    """

    # Split windows
    windowed = np.lib.stride_tricks.sliding_window_view(
        conv_matrix,
        window_shape=(filters_matrix.shape[1], filters_matrix.shape[2]),
        axis=(2, 3)
    )

    # perform multiplication using einstein notation
    return np.einsum('albcjk,ijkl->aibc', windowed, filters_matrix, optimize=True)


def max_pool(matrix, pool):
    """
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
    x[x < 0] = 0
    return 1


def dense_layers(flat_input_shape, nodes):
    """
    Expects 3d input_matrix
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
    return np.exp(output_array) / np.sum(np.exp(output_array), axis=1)


def normalize_features(input_matrix):
    for i in range(input_matrix.shape[0]):
        input_matrix[i] = (input_matrix[i] - np.mean(input_matrix[i])) / np.std(input_matrix[i])
    return 1
