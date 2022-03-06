from functions.backProp import *
from functions.forwardProp import *


class CNN:

    # Initialize class by choosing by choosing parameters
    def __init__(
        self, x_train, y_train, n_filters=(32, 64), filter_sizes=((3, 3), (3, 3)),
        pool_sizes=((2, 2),), dense_nodes=(128, 10), img_features=1,
        n_activations=3, eta=0.001, epochs=10
    ):

        self.x_train = x_train
        normalize_features(self.x_train)
        self.y_train = y_train
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.pool_sizes = pool_sizes
        self.dense_nodes = dense_nodes
        self.features = img_features
        self.n_activations = n_activations
        # self.batch_size = batch_size
        self.batch_size = self.x_train.shape[0]
        self.eta = eta
        self.epochs = epochs

        # 1.1 CALCULATE FINAL SHAPE FOR THE DENSE LAYERS
        res_shape = list(self.x_train.shape)
        res_shape = [
            self.batch_size,
            self.n_filters[0],
            res_shape[1] - self.filter_sizes[0][0] + 1,
            res_shape[2] - self.filter_sizes[0][1] + 1
        ]
        res_shape = [
            self.batch_size,
            self.n_filters[1],
            res_shape[2] - self.filter_sizes[1][0] + 1,
            res_shape[3] - self.filter_sizes[1][1] + 1
        ]
        res_shape = [
            self.batch_size,
            res_shape[1],
            # If width and height are divisible over pool sizes - don't add anything
            # If not perfectly divisible - add 1 to each dimension
            res_shape[2] // self.pool_sizes[0][0] + int(res_shape[2] % self.pool_sizes[0][0] != 0),
            res_shape[3] // self.pool_sizes[0][1] + int(res_shape[3] % self.pool_sizes[0][1] != 0)
        ]
        self.pooled_shape = res_shape

        # 1.2 INITIALIZE WEIGHT MATRICES
        # Dense layer initialized with Kaiming He method
        weights = [
            random_filters(self.n_filters[0], self.filter_sizes[0][0], self.features) * \
            np.sqrt(2 / np.prod(res_shape[1:])),
            random_filters(self.n_filters[1], self.filter_sizes[1][0], self.n_filters[0]) * \
            np.sqrt(2 / np.prod([self.filter_sizes[1][0], self.filter_sizes[1][1], self.features]))
        ] + [
          np.random.randn(np.prod(res_shape[1:]), self.dense_nodes[0]) * np.sqrt(
              2 / np.prod(res_shape[1:])),
          np.random.randn(*self.dense_nodes) * np.sqrt(2 / self.dense_nodes[0])
        ]

        self.weights = weights

        # 1.3 INITIALIZE DERIVATIVES MATRICES (should be zeros)
        derivatives = []
        for l in range(len(self.weights)):
            derivatives.append(np.zeros(self.weights[l].shape))

        self.derivatives = derivatives

        # 1.4 INITIALIZE PRE-ACTIVATION and ACTIVATION MATRICES (zeros)
        activations = []
        for a in range(len(self.n_filters)):
            activations.append(np.zeros(self.weights[a].shape))
        activations.append(np.zeros((1, np.prod(res_shape[1:]))))
        for a in range(len(self.dense_nodes)):
            activations.append(np.zeros((self.batch_size, self.dense_nodes[a])))
        self.activations = activations

        self.pre_act = activations.copy()

    # 2. Forward propagate and calculate error
    def forward_propagate(self):

        # Load input data from the class
        # print(self.x_train.shape, self.x_train.min(), self.x_train.max())

        # 1st convolution
        res = conv1(self.x_train, self.weights[0])
        self.pre_act[0] = res
        relu_fun(res)
        self.activations[0] = res
        # print("First conv: ", res.shape)
        # print("AFTER 1ST CONV: ", res.min(), res.max())

        # 2nd convolution
        res = conv2(res, self.weights[1])
        self.pre_act[1] = res
        relu_fun(res)
        self.activations[1] = res
        # print("2nd CONV: ", res.shape)
        # print("AFTER 2ND CONV: ", res.min(), res.max())

        # MaxPool
        res = max_pool(res, (2, 2))
        # print(res.shape)
        # print("AFTER POOLING: ", res.min(), res.max())

        # Flatten
        res = res.reshape(res.shape[0], -1)
        self.activations[2] = res
        # print("FLATTEN: ", res.shape)

        # 1st Dense
        res = res.dot(self.weights[2])
        self.pre_act[2] = res
        relu_fun(res)
        self.activations[3] = res
        # print(res.shape)
        # print("AFTER 1ST DENSE: ", res.min(), res.max())

        # 2nd Dense and softmax
        res = res.dot(self.weights[3])
        self.pre_act[3] = res
        # print(res.shape)
        # normalize_features(res)
        res = softmax_fun(res)
        self.activations[4] = res
        # print("AFTER 1ST CONV: ", res.min(), res.max())

        return 1

    # 3. Back propagate error
    def backprop(self):

        # DENSE LAYERS DERIVATIONS
        # 2nd DENSE (w3) derivation
        delta = self.activations[4] - self.y_train[:1]
        # print(delta.shape)
        self.derivatives[3] = delta.T.dot(self.activations[3]).T
        # print("dlw3: ", dlw3.T.shape)

        # 1st DENSE (w2) derivation
        delta = delta.dot(self.weights[3].T).T * relu_prime(self.pre_act[2]).T
        # print(delta.shape)
        self.derivatives[2] = delta.dot(self.activations[2]).T
        # print("dlw2: ", dlw2.shape)
        # print(delta.shape)

        # MAX POOL DERIVATION
        delta = delta.T.dot(self.weights[2].T).reshape(self.pooled_shape)

        # Stretch delta so that it corresponds to C2 output
        # https://medium.com/the-bioinformatics-press/only-numpy-understanding-back-propagation-for-max-pooling-layer-in-multi-layer-cnn-with-example-f7be891ee4b4
        delta = np.repeat(
            np.repeat(
                delta, self.pool_sizes[0][1], axis=3
            ), self.pool_sizes[0][0], axis=2
        )

        # MaxPool derivative over C2 output
        # = Matrix of 0s and 1s (maximum values in a pooling window and values that are not maximums)
        # OUTPUT: Cell-wise multiplication of delta, and MaxPool derivative over C2 output
        delta = delta * max_pool_prime(self.activations[1], self.pool_sizes[0])
        delta = delta * relu_prime(self.pre_act[1])

        # C2 (w1) DERIVATION
        self.derivatives[1] = c2fprime(self.activations[0], delta)
        # print("dlw1: ", dlw1.shape)

        # C1 (w0) DERIVATION
        delta = c2xprime(self.weights[1], delta)
        delta = delta * relu_prime(self.pre_act[0])
        self.derivatives[0] = c1fprime(self.x_train, delta)
        # print("dlw0: ", dlw0.shape)
        return 1

    # 4. Update weight matrices
    def update_weights(self):
        for w in range(len(self.weights)):
            self.weights[w] = self.weights[w] - self.eta * self.derivatives[w]
        return 1

    # Train function
    def train(self):

        # for loop for training set
        losses = []
        for e in range(self.epochs):
            self.forward_propagate()
            # print(self.activations[-1][0])
            loss = -np.log(
                self.activations[-1][0][
                    np.where(self.y_train[:1][0])[0][0]
                ]
            )
            losses.append(loss)
            print(f"EPOCH {e}:", loss)
            self.backprop()
            self.update_weights()

        # for loop for validation fraction of train set

        return 1
