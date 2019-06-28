import numpy as np
#import tweepy
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Activation, Reshape
from keras.layers import LSTM, Dropout, Embedding
from model import Model
import time


# auth=tweepy.OAuthHandler("","")
# auth.set_access_token("","")
# api=tweepy.API(auth)

def generate_model(n_vocab, seq_len ):

    model = Sequential()
    layer = Model.lstm_layer
    input_shape = (1, seq_len, n_vocab)
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


with open("LaxmiPrasadDevkotaPoem.txt", "r", encoding='utf8', errors='ignore') as f:
    corpus = f.read()

chars = sorted(list(set(corpus)))
encoding = {c: i for i, c in enumerate(chars)}
decoding = {i: c for i, c in enumerate(chars)}
print("loaded encoding and decoding data sets")

sentence_length= Model.seq_length

with open ("model.yaml",'r') as f:
    yaml_string=f.read()

#model= model_from_yaml(yaml_string)
model = generate_model(len(chars), Model.seq_length)
model.load_weights("model/weights-19-1.424.hdf5")
seed_starting_index = np.random.randint(0, len(corpus) - sentence_length)
seed_sentence = corpus[seed_starting_index:seed_starting_index + sentence_length]
X_predict = np.zeros((1, sentence_length, len(chars)), dtype=np.bool)
for i, char in enumerate(seed_sentence):
    X_predict[0, i, encoding[char]] = 1


generated = ""
for i in range(200): # longer prediction greater range
    prediction = np.argmax(model.predict(X_predict))
    generated += decoding[prediction]
    activations = np.zeros((1, 1, len(chars)), dtype=np.bool)
    activations[0, 0, prediction] = 1
    X_predict = np.concatenate((X_predict[:, 1:, :], activations), axis=1)

print(seed_sentence +"\n"+ generated)
# api.update_status(seed_sentence + generated)
# print("Posted Tweet")