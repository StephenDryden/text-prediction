#!/usr/bin/python3/
import pickle

import tensorflow as tf
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

# load notes.csv
text_df = pd.read_csv("notes.csv")

# load just the Notes column and combine all values into a single string
text = list(text_df.Notes)
joined_text = " ".join(text)

# load the first x characters
partial_text = joined_text[:100000]

# tokenize the text data, convert into a list of words (tokens)
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())

# create a dictionary of unique tokens, mapping each unique token to a numerical index
unique_tokens = np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}

# prepare input data for neural network by creating sequences of n_words length and their corresponding next words
n_words = 7
input_words = []
next_words = []

for i in range(len(tokens) - n_words) :
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])


# perform 'one-hot encoding' on input sequences and target next words
# One-hot encoding is a way to represent categorical data (in this case, words) as binary vectors.
# Each unique word in the vocabulary is assigned a unique index, and for each word in the input sequence or target word,
# a vector is created with a 1 at the index corresponding to that word, and 0s everywhere else.
x = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
y = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)

for i , words in enumerate(input_words):
    for j, word in enumerate(words):
        x[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_words[i]]] = 1


# define a LSTM model with two LSTM layers, a Dense layer and a Softmax activation layer
# Define the input shape
input_shape = (n_words, len(unique_tokens))

# Create an Input layer
inputs = tf.keras.layers.Input(shape=input_shape)

# Define the model
model = tf.keras.models.Sequential()
model.add(inputs)
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dense(len(unique_tokens)))
model.add(tf.keras.layers.Activation("softmax"))

# compile the model with a categorical cross-entropy loss and RMSprop
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), metrics=["accuracy"])
# train the model on the prepared x and y data for x epochs
model.fit(x,y, batch_size=128, epochs=30, shuffle=True)

# save and load - technically don't need to do this as already in memory
model.save("notes.keras")

with open("unique_tokens.pkl", "wb") as f:
    pickle.dump(unique_tokens, f)

with open("unique_token_index.pkl", "wb") as f:
    pickle.dump(unique_token_index, f)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

def load_model():
    # Load the trained model
    model = load_model("notes.keras")

    # Load other necessary variables
    with open("unique_tokens.pkl", "rb") as f:
        unique_tokens = pickle.load(f)
    with open("unique_token_index.pkl", "rb") as f:
        unique_token_index = pickle.load(f)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    n_words = 7  # or whatever value you used during training

    return model, unique_tokens, unique_token_index, tokenizer, n_words
