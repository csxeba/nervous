from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization, Add
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical, normalize, plot_model

from nervous import StressedNet
from nervous.utility.config import StressedNetConfig


class Data:

    def __init__(self):
        (X, Y), (tX, tY) = mnist.load_data()
        self.X, self.tX = map(normalize, [X, tX])
        self.Y, self.tY = map(lambda y: to_categorical(y, num_classes=10), [Y, tY])

    def get_iterator(self, batch_size=32):
        while 1:
            for start in range(0, len(self.X)-batch_size-1, batch_size):
                yield self.X[start:start+batch_size], self.Y[start:start+batch_size]

    @property
    def testing(self):
        return self.tX, self.tY


def build_small_mlp(input_shape, output_shape):
    inputs = Input(input_shape)
    x = Flatten()(inputs)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(output_shape[0], activation="softmax")(x)
    ann = Model(inputs, x)
    ann.compile(SGD(), loss="categorical_crossentropy", metrics=["acc"])
    return ann


def build_x_mlp(input_shape, output_shape):
    inputs = Input(input_shape)
    x = Flatten()(inputs)
    x1 = Dense(128, activation="relu")(x)
    x2 = Dense(128, activation="tanh")(x)
    cross = Dense(256, activation="tanh")
    x11 = cross(x1)
    x22 = cross(x2)
    x111 = Dense(128, activation="relu")(x11)
    x222 = Dense(128, activation="tanh")(x22)
    x1111 = Add()([x1, x111])
    x2222 = Add()([x2, x222])
    x3 = Add()([x1111, x2222])
    x4 = Dense(output_shape[0], activation="softmax")(x3)
    ann = Model(inputs, x4)
    ann.compile(SGD(), loss="categorical_crossentropy", metrics=["acc"])
    plot_model(ann)
    return ann


def main():
    dataset = Data()
    ann = build_small_mlp(dataset.X.shape[1:], output_shape=dataset.Y.shape[1:])
    print("Pretraining MLP")
    ann.fit_generator(dataset.get_iterator(batch_size=32), steps_per_epoch=10, epochs=3, verbose=2,
                      validation_data=dataset.testing)
    stressed = StressedNet(ann, StressedNetConfig(synaptic_environmental_constraint=0.6,
                                                  group_environmental_constraint=0.6,
                                                  stress_factor=0.8,
                                                  save_folder="/data/models/stressednet/"))
    stressed.fit_generator(dataset.get_iterator(batch_size=32), generations=3, num_offsprings=1,
                           steps_per_epoch=10, epochs=3, verbose=2, validation_data=dataset.testing)


if __name__ == '__main__':
    main()
