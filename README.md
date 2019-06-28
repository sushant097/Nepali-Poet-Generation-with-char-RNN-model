# Nepali Poem Generation using char-rnn  model
> Inspired By [Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) work.

**Description:**
This project use the char-rnn model where
* layer of LSTM =3 with units=256
* Dropout Probability = 0.2
* Final Dense layer with units = Number of vocabs=90

Here i use stateful=true in each LSTM layer since it help to maintain long term dependencies in each sequence. If `stateful=false` you can use whole file at a once to train.

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
train.py  ==> Used to train the model. 
Here due to `stateful=true` in lSTM, it should be follow as train in Batch.
```

> This project you can also use with English text Generation. For eg: Joke large text > 1MB data to generate the new Joke text based on Char-RNN model.

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

**Predicted Text**
```markdown
﻿की
खरिद प्यार मासुकी !
ए गर्त ! यौनप्यासकी !
ए पुत्तली विनाश
 रेशम चुल्यो !
बादल भन्छौ तिमी नै बिरामी,
आत्मा बस्छन् किन ? प्राण !
निश्चल, स्वर्ग भै सारा !

नेपालीको कालो कुस्की, त्यो प्राण !
के तिम्रो सुनको र म पापी भन्छन् ।
बिरामी
or 
कसरी गइन् ती मेरी आमा, तिम्लाई छाडेर,
कसरी गइन् बिचरीलाई
सुनको बारी,

मुनाको बाटो बुझेका आँखा रसाई बिरामी,
कुइरी दिदी ! चीतनले भन्छिन्, “दैवले हुँदैन,

म आएँ आए समय निदाइ सुनको रामा ।

कलिलो बस्छौ सारा भएको बेलामा बल्दछ,
हे मेरी आमा ! म हाम्रा मेरी ! म आएँ

```
**See on the predicted.txt file for more**


### LOSS AND Accuracy Overview
![accuracy](https://github.com/sushant097/Nepali-Poet-Generation-with-char-RNN-model/blob/master/image/accuracy.png)
![Loss](https://github.com/sushant097/Nepali-Poet-Generation-with-char-RNN-model/blob/master/image/loss.png)


<i>The model is run for only 21 epoch take me 5 hours in 940MX 2GB Nvidia GPU. The loss and accuracy untill 21 epochs is:
Loss:1.364  , Accuracy=61%. Train it for more than 40 epochs as loss is nearly 0.20 that can give perfect output of the model.</i>

**Tips to Increase Accuracy of the Model**
* The dataset should be well formated and arranged.
* Increase Size of dataset minimum >1MB of text .
* Use More Layer of `LSTM` to deeper understanding but aware of model overfitting.
* Use more Epoch untill loss is very minimum.
