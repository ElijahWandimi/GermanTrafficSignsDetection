from numpy.core.fromnumeric import shape
import tensorflow
from tensorflow import keras
from tensorflow.python.ops.gradients_util import _Inputs

# functional approach for the  model -CNN to rain on the images

def StreetSignDetector(num_classes):
    model_input = keras.layers.Input(shape=(60, 60, 3))

    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(model_input)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(512, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.BatchNormalization()(x)

    # x = keras.layers.Flatten()(x)
    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs=model_input, outputs=x)


if __name__ == '__main__':
    model = StreetSignDetector(43)
    model.summary()