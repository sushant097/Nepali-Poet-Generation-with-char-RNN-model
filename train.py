import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard

from model import generate_model, Model, save_weights

import argparse
import os
import time

LOG_DIR = '../logs'


class TrainLogger(object):
    def __init__(self, file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        s = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)


with open("LaxmiPrasadDevkotaPoem.txt", "r", encoding='utf8', errors='ignore') as f:
    corpus = f.read()

char = sorted(list(set(corpus)))  ## len(chars) is n_vocab
with open('char.txt', 'w', encoding='utf8') as f:
    f.write(str(char))

encoding = {c: i for i, c in enumerate(char)}
decoding = {i: c for i, c in enumerate(char)}

print(encoding)
print(decoding)
print("Unique characters " + str(len(char)))
print("Total characters " + str(len(corpus)))

sentence_length = Model.seq_length
step = 1
sentences = []
nextchar = []

for i in range(0, len(corpus) - sentence_length, step):
    sentences.append(corpus[i:i + sentence_length])
    nextchar.append(corpus[i + sentence_length])

print("Train set length " + str(len(nextchar)))

x = np.zeros((len(sentences), sentence_length, len(char)), dtype=np.bool)
y = np.zeros((len(sentences), len(char)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, encoding[character]] = 1
    y[i, encoding[nextchar[i]]] = 1

print("Training set(X) shape" + str(x.shape))
print("Training set(Y) shape" + str(y.shape))

BATCH_SIZE = Model.batch_size
SEQ_LENGTH = len(char)


def read_batches():
    cur_index = 0
    steps_per_epoch = len(x) // BATCH_SIZE
    while True:
        if(cur_index + BATCH_SIZE) > len(x):
            cur_index = 0
        batchRange = range(cur_index, cur_index + BATCH_SIZE)
        X = np.asarray([x[i,:,:] for i in batchRange])
        Y = np.asarray([y[i, :] for i in batchRange])
        cur_index += BATCH_SIZE
        yield X, Y


def load_trained_model(weights_path):
    model = generate_model(len(char), sentence_length)
    model.load_weights(weights_path)
    print("=================================================")
    print("Model restore from : {}".format(weights_path))
    print("=================================================")
    return model


def train_model(epochs=40, save_freq=5, load_model=False):

    if load_model:
        model = load_trained_model("model/weights-19-1.424.hdf5")

    else:
        model = generate_model(len(char), sentence_length)
        architecture = model.to_yaml()
        with open('model.yaml', 'a') as model_file:
            model_file.write(architecture)
    print(model.summary())
    tensorboard = TensorBoard(log_dir="logs", write_graph=True, write_images=True)
    steps_per_epoch = len(x) // BATCH_SIZE
    file_path = "model/weights-{epoch:02d}-{loss:.3f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor="loss", verbose=1, save_best_only=True, mode="min")

    model.fit_generator(
        generator=read_batches(),
        steps_per_epoch=steps_per_epoch,
        verbose=2,  # 0 = silent, 1 = progress bar, 2 = one line per epoch.
        epochs=epochs,  # stopEpoch,
        initial_epoch=19,
        validation_data=None,
        #  validation_steps=int(50 // self.batchSize),   # len(self.loader.validationSamples) // ModelTrain.batchSize
        callbacks=[checkpoint, tensorboard]
        #max_queue_size=1
    )


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Train the model on some text.')
    # parser.add_argument('--input', default='input.txt', help='name of the text file to train from')
    # parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train for')
    # parser.add_argument('--freq', type=int, default=5, help='checkpoint save frequency')
    # args = parser.parse_args()

    train_model(load_model=True)
