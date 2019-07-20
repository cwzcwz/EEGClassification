import keras
from keras.models import Sequential


def eegnet(
        num_eegnet_classes=2,
):
    """construct a eegnet model
        Returns
        A keras.models.Model which takes an image as input and outputs the classification on the image.
        The outputs is defined as follow:
            eegnet_classification
    """

    model = Sequential()
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=400),
                                         input_shape=(40000, 8)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(num_eegnet_classes, activation='softmax'))
    return model
