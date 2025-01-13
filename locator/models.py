"""Neural network model definitions"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np


def euclidean_distance_loss(y_true, y_pred):
    """Custom loss function using Euclidean distance"""
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def create_network(input_shape, width=256, n_layers=8, dropout_prop=0.25):
    """Create the neural network model"""
    model = keras.Sequential()
    model.add(layers.BatchNormalization(input_shape=(input_shape,)))

    # First half of layers
    for i in range(int(np.floor(n_layers / 2))):
        model.add(layers.Dense(width, activation="elu"))

    # Middle dropout layer
    model.add(layers.Dropout(dropout_prop))

    # Second half of layers
    for i in range(int(np.ceil(n_layers / 2))):
        model.add(layers.Dense(width, activation="elu"))

    # Two final coordinate prediction layers
    model.add(layers.Dense(2))
    model.add(layers.Dense(2))

    model.compile(optimizer="Adam", loss=euclidean_distance_loss)

    return model
