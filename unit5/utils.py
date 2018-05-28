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

	text = simplify_text(text)

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
	return x, y

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