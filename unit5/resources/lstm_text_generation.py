'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import os
import numpy as np
import random
import sys
import io
import argparse
import json
from collections import defaultdict

experiment_name = "128 to 64 hidden units large dataset with simplification"

parser = argparse.ArgumentParser()
parser.add_argument("--cont", action="store_true", default=False)
args = parser.parse_args()



outputs_directory = 'outputs'
if not os.path.exists(outputs_directory):
        os.makedirs(outputs_directory)

def simplify_text(text):
    counts = defaultdict(int)
    for ch in text:
        counts[ch] += 1
    print(len(counts), "keys before simplification")
    counts = [(counts[k], k) for k in counts.keys()]
    print(sorted(counts, reverse=True))
    removed = 0
    for count, ch in counts:
        if count <= 200:
            text = text.replace(ch, '')
            removed += 1
    print(removed, "characters removed")
    return text

path = 'fake.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()

text = simplify_text(text)

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
print(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant chunks of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


if not args.cont:
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    with open(outputs_directory + "/logs.json", 'w') as f:
        json.dump({"experiment_name": experiment_name, "logs": []}, f)
else:
    model_file = sorted(os.listdir(outputs_directory), reverse=True)[0]
    model = load_model(outputs_directory + "/" + model_file)
    print("Loading from", model_file)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    # TODO remove
    with open(outputs_directory + "/logs.json", 'r') as f:
        curr_dict = json.load(f)
    curr_dict['logs'].append(logs)
    with open(outputs_directory + "/logs.json", 'w') as f:
        json.dump(curr_dict, f)
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    model.save(outputs_directory + "/lstm_epoch%d.h5" % epoch)
    model.save(outputs_directory + "/logs_epoch%d.h5" % epoch)


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=300,
          callbacks=[print_callback])
