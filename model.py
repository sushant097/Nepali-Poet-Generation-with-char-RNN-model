from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Activation, Reshape
from keras.layers import LSTM, Dropout, Embedding
import os


class Model:
    batch_size = 128
    lstm_layer = 3
    drop_prob = 0.2
    model_dir = "model/"
    seq_length = 60


def save_weights(epoch, model):
    MODEL_DIR = Model.model_dir
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))


def generate_model(n_vocab, seq_len ):

    model = Sequential()
    layer = Model.lstm_layer
    input_shape = (Model.batch_size, seq_len, n_vocab)
    for i in range(layer):
        if i == 0:
            # add first hidden layer - This crashes if num_layers == 1
            model.add(LSTM(256, batch_input_shape=input_shape,return_sequences=True, stateful=True))
        elif i == layer-1:
            model.add(LSTM(256, return_sequences=False, stateful=True))
        else:
            model.add(LSTM(256, return_sequences=True, stateful=True))

        model.add(Dropout(Model.drop_prob))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

