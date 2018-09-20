from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical, normalize

from nervous import StressedNet


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
    x = Dense(512, activation="tanh")(x)
    x = BatchNormalization()(x)
    x = Dense(output_shape[0], activation="softmax")(x)
    ann = Model(inputs, x)
    ann.compile(SGD(), loss="categorical_crossentropy", metrics=["acc"])
    return ann


def main():
    dataset = Data()
    ann = build_small_mlp(dataset.X.shape[1:], output_shape=dataset.Y.shape[1:])
    print("Pretraining MLP")
    ann.fit_generator(dataset.get_iterator(batch_size=32), steps_per_epoch=100, epochs=10, verbose=2,
                      validation_data=dataset.testing)
    stressed = StressedNet(ann, StressedNet.Config(synaptic_normalizing_term=100,
                                                   group_normalizing_term=100,
                                                   synaptic_environmental_constraint=0.8,
                                                   group_environmental_constraint=0.8,
                                                   save_folder="/data/models/stressednet/"))
    offsprings = stressed.fit_generator(dataset.get_iterator(batch_size=32), generations=10, num_offsprings=1,
                                        steps_per_epoch=100, epochs=10, verbose=2, validation_data=dataset.testing)
    for i, generation in enumerate(offsprings, start=1):
        print("Generation", i)
        for j, offspring in enumerate(offsprings, start=1):
            offspring.describe()


if __name__ == '__main__':
    main()
