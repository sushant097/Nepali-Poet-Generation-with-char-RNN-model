# Nepali Poem Generation using char-rnn  model
> Inspired By Andrej Karpathy work 
**Description:**
This project use the char-rnn model where
* layer of LSTM =3 with units=256
* Dropout Probability = 0.2
* Final Dense layer with units = Number of vocabs=90

Here i use stateful=true in each LSTM layer since it help to maintain long term dependencies in each sequence.

```python
BATCH_SIZE = 128
# to read batches created
def read_batches():
    cur_index = 0
    steps_per_epoch = len(x) // BATCH_SIZE  # x is whole text data in a vector
    while True:
        if(cur_index + BATCH_SIZE) > len(x):
            cur_index = 0
        batchRange = range(cur_index, cur_index + BATCH_SIZE)
        X = np.asarray([x[i,:,:] for i in batchRange])
        Y = np.asarray([y[i, :] for i in batchRange])
        cur_index += BATCH_SIZE
        yield X, Y
```
The following 3 files are used mainly as:
```markdown
generate.py ==> Used to generate the predicted new text of the given length i.e in this case is Nepali Poem.
model.py ==> Used to create the model of the Network
train.py  ==> Used to train the model. Here due to **`stateful=true`** in lSTM, it should be follow as train in Batch.
```
The Model Summary in Keras is shown below:
```markdown
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (128, 60, 256)            355328
_________________________________________________________________
dropout_1 (Dropout)          (128, 60, 256)            0
_________________________________________________________________
lstm_2 (LSTM)                (128, 60, 256)            525312
_________________________________________________________________
dropout_2 (Dropout)          (128, 60, 256)            0
_________________________________________________________________
lstm_3 (LSTM)                (128, 256)                525312
_________________________________________________________________
dropout_3 (Dropout)          (128, 256)                0
_________________________________________________________________
dense_1 (Dense)              (128, 90)                 23130
=================================================================
Total params: 1,429,082
Trainable params: 1,429,082
Non-trainable params: 0
```
You can also use Embedding layer on top of LSTM as embedding every character as Dense Layer to improve accuracy but not much seem as improvement. Like this:
```python
 model.add(Embedding(vocab_size, units, batch_input_shape=(batch_size, seq_len)))
```
The char-RNN model is here used to generate the Nepali Poem written by Adikabi "Laxmi Prasad Devkota" who is pioneer in Nepali Literature.
* n_vocabs = number of unique characters appear in the whole text i.e 90
* seq_length = sentence length to be taken set as 60
* Batch_size = the number of sentence text that is pass to the model at once i.e set as 128. 
