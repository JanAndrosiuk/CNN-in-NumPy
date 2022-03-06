from functions.backProp import *
from functions.forwardProp import *
from tqdm import tqdm


class CNN:

    # Initialize class by choosing by choosing parameters
    def __init__(
        # x_train and y_train are added here only to get weight matrices shapes,
        # those values won't be forward or backward propagated
        self, x_train, y_train, n_filters=(32, 64), filter_sizes=((3, 3), (3, 3)),
        pool_sizes=((2, 2),), dense_nodes=(128, 10), img_features=1,
        n_activations=3, eta=0.001, epochs=10, use_oe=True
    ):

        self.x_train = x_train
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
        self.use_oe = use_oe

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
    def forward_propagate(self, img_array):

        # Normalize input data
        normalize_features(img_array)
        # print(img_array.shape, img_array.min(), img_array.max())

        # 1st convolution
        res = conv1(img_array, self.weights[0], self.use_oe)
        self.pre_act[0] = res
        relu_fun(res)
        self.activations[0] = res
        # print("First conv: ", res.shape)
        # print("AFTER 1ST CONV: ", res.min(), res.max())

        # 2nd convolution
        res = conv2(res, self.weights[1], self.use_oe)
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
    def backward_propagate(self, inp_array, out_array):

        # DENSE LAYERS DERIVATIONS
        # 2nd DENSE (w3) derivation
        # delta = self.activations[4] - self.y_train[:1]
        delta = self.activations[4] - out_array
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
        self.derivatives[1] = c2fprime(self.activations[0], delta, self.use_oe)
        # print("dlw1: ", dlw1.shape)

        # C1 (w0) DERIVATION
        delta = c2xprime(self.weights[1], delta, self.use_oe)
        delta = delta * relu_prime(self.pre_act[0])
        self.derivatives[0] = c1fprime(inp_array, delta, self.use_oe)
        # print("dlw0: ", dlw0.shape)
        return 1

    # 4. Update weight matrices
    def update_weights(self):
        for w in range(len(self.weights)):
            self.weights[w] = self.weights[w] - self.eta * self.derivatives[w]
        return 1

    # Train function
    def train(self, inp_array, out_array, train_val_split=0.8):

        # Split into train and validation sets
        train_val_size = int(np.floor(train_val_split * inp_array.shape[0]))
        train_x, val_x = inp_array[:train_val_size, :], inp_array[train_val_size:, :]
        train_y, val_y = out_array[:train_val_size, :], inp_array[train_val_size:, :]

        # Store the history of training over each epoch
        losses = np.zeros((self.epochs, 2))
        losses.fill(np.nan)

        for e in range(self.epochs):

            # 1. Train Loop
            loss_train = 0
            for i in tqdm(range(train_x.shape[0])):
                # print(train_x[i].shape, train_y[i].shape)
                self.forward_propagate(np.expand_dims(train_x[i], 0))
                loss_i = -np.log(
                    self.activations[-1][0][
                        np.where(train_y[i])[0][0]
                    ]
                )
                # print(f"TRAINING SAMPLE {i} LOSS: {loss_i}")

                loss_train += loss_i
                self.backward_propagate(
                    np.expand_dims(train_x[i], 0),
                    np.expand_dims(train_y[i], 0)
                )
                self.update_weights()
            losses[e, 0] = loss_train / train_x.shape[0]

            # 2. Validation Loop
            loss_val = 0
            for j in tqdm(range(val_x.shape[0])):
                self.forward_propagate(np.expand_dims(val_x[j], 0))
                loss_j = -np.log(
                    self.activations[-1][0][
                        np.where(val_y[j])[0][0]
                    ]
                )
                # print(f"VAL SAMPLE {i} LOSS: {loss_i}")
                loss_val += loss_j
            losses[e, 1] = loss_val / val_x.shape[0]

            # print(f"EPOCH {e}, AVERAGE TRAIN LOSS: {loss_train/train_x.shape[0]}")

        return losses

    # Make predictions on test set
    def predict(self):
        pass
