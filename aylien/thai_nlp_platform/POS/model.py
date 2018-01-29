"""
Keras Model
"""

from keras.models import Sequential
from keras.layers import Embedding, LSTM, TimeDistributed, Dense, Dropout, Bidirectional
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam

from thai_nlp_platform.POS import constant


class Model(object):
    def __init__(self, num_step):

        model = Sequential()
        model.add(Bidirectional(LSTM(32, return_sequences=True),
                       input_shape=(num_step, constant.DATA_DIM)))
        model.add(Dense(32, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        self.model = model
