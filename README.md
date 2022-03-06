# CNN-in-NumPy
The project serves a purpose of demystifying internal CNN mechanisms of forward and back propagation. 

For this purpose we used multiple NumPy functions to train the CNN model on keras MNIST dataset.

The idea was to (+/-) replicate the tensorflows model of the given pipeline:

```python
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
        activation="relu", input_shape=(28, 28, 1))
    )
    model.add(
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
        activation="relu")
    )
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
              
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adadelta(learning_rate=1),
        metrics='accuracy'
    )
```
## Simplifications / constraints:

1. The model uses simple gradient descent optimizer


## Future improvements
- [ ] Including bias matrices
- [ ] Including batch size > 1
- [ ] Including easier modification of parameters such as number of hidden layers / number of convolution - maxpool sequences
- [ ] Further optimization of functions

Collaborators:
* Gregor Baer [[github]](https://github.com/gregorbaer)
* Marius Lupulescu [[github]](https://github.com/mariusadrian77)
* Lachezar Popov


