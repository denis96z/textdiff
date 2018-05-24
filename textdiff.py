import numpy as np
import random as rnd

from keras import Sequential, Input, Model
from keras.layers import Dense, \
    Activation, Dropout, Embedding, \
    Flatten, LSTM, TimeDistributed, Add
from keras.optimizers import Adagrad

WORD_LEN = 100
SENTENCE_LEN = 5

NUM_SENTENCES_1 = 3
NUM_SENTENCES_2 = 5


def main():
    model = create_model()
    x1, x2, y = get_random_training_set(10000)
    model.fit([x1, x2], y, epochs=500, validation_split=0.2, verbose=1)


def create_model():
    x1_in, x1_out = create_model_common_parts()
    x2_in, x2_out = create_model_common_parts()
    x3 = Add()([x1_out, x2_out])
    out = Dense(1)(x3)
    model = Model(inputs=[x1_in, x2_in], outputs=out)
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model


def create_model_common_parts():
    x0 = Input(shape=(SENTENCE_LEN, WORD_LEN,))
    x1 = Dense(64, activation='relu')(x0)
    x2 = Dense(100, activation='relu')(x1)
    x3 = Dropout(0.5)(x2)
    x4 = Flatten()(x3)
    return x0, x4


def create_recurrent_model():
    model = Sequential()
    model.add(Dense(256, input_shape=(SENTENCE_LEN, WORD_LEN,)))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(SENTENCE_LEN, ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Embedding(WORD_LEN, 256))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(WORD_LEN)))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='mse',
                  optimizer=Adagrad(), metrics=['accuracy'])
    return model

def get_random_training_set(num_sentences):
    x1 = np.array([get_random_sentence() for _ in range(num_sentences)])
    x2 = np.array([get_random_sentence() for _ in range(num_sentences)])
    y = np.array([rnd.gauss(0.5, 0.45) for _ in range(num_sentences)])
    return x1, x2, y


def get_random_sentence():
    return np.array([get_random_word() for _ in range(SENTENCE_LEN)])


def get_random_word():
    return np.array([rnd.gauss(0.0, 0.95) for _ in range(WORD_LEN)])


if __name__ == '__main__':
    main()
