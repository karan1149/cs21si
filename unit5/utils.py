import os
import numpy as np
import random
import sys
import io
import argparse
import json
from collections import defaultdict

def get_X_y(text):
	text = text.lower()
	print(len(text))
	text = simplify_text(text)
	print(len(text))

	print('Corpus length:', len(text))

	chars = sorted(list(set(text)))
	print('Total chars:', len(chars))
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
	print('Chunk length:', maxlen)
	print('Number of chunks:', len(sentences))

	x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
	y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
	for i, sentence in enumerate(sentences):
	    for t, char in enumerate(sentence):
	        x[i, t, char_indices[char]] = 1
	    y[i, char_indices[next_chars[i]]] = 1
	return x, y, char_indices, indices_char

def simplify_text(text):
    counts = defaultdict(int)
    for ch in text:
        counts[ch] += 1
    counts = [(counts[k], k) for k in counts.keys()]
    removed = 0
    for count, ch in counts:
        if count <= 200:
            text = text.replace(ch, '')
            removed += 1
    return text

def sample_from_model(model, text, char_indices, indices_char, chunk_length, number_of_characters, seed=""):
	text = text.lower()
	start_index = random.randint(0, len(text) - chunk_length - 1)
	for diversity in [0.2, 0.5, 0.7]:
	    print('----- diversity:', diversity)

	    generated = ''
	    if not seed:
	    	sentence = text[start_index: start_index + chunk_length]
	    else:
	    	seed = seed.lower()
	    	sentence = seed[:chunk_length]
	    	sentence = ' ' * (chunk_length - len(sentence)) + sentence
	    generated += sentence
	    print('----- Generating with seed: "' + sentence + '"')
	    sys.stdout.write(generated)

	    for i in range(400):
	        x_pred = np.zeros((1, chunk_length, number_of_characters))
	        for t, char in enumerate(sentence):
	            x_pred[0, t, char_indices[char]] = 1.

	        preds = model.predict(x_pred, verbose=0)[0]
	        next_index = sample(preds, diversity)
	        next_char = indices_char[next_index]

	        generated += next_char
	        sentence = sentence[1:] + next_char

	        sys.stdout.write(next_char)
	        sys.stdout.flush()
	    print("\n")

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)