#!/usr/bin/python3/
import random
import pickle

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop


# load notes.csv
text_df = pd.read_csv("notes.csv")

# load just the Notes column and combine all values, joining with a space
text = list(text_df.Notes)
joined_text = " ".join(text)

# load the first x characters
partial_text = joined_text[:10000]

# sanatize data
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())

unique_tokens = np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}

# take x words before prediction
n_words = 7
input_words = []
next_words = []

for i in range(len(tokens) - n_words) :
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])

x = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
y = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)

for i , words in enumerate(input_words):
    for j, word in enumerate(words):
        x[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_words[i]]] = 1


# define the model
model = Sequential()
model.add(LSTM(128, input_shape=(n_words, len(unique_tokens)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))

# compile the model
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
# increase epochs to train for longer
model.fit(x,y, batch_size=128, epochs=30, shuffle=True)

# save and load - technically don't need to do this as already in memory
model.save("notes.h5")
model = load_model("notes.h5")

# function used to predict the next word
def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    x = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        x[0,i,unique_token_index[word]] = 1

    predictions = model.predict(x)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

# example of word prediction, given x words give me 5 suggestions
# possible = predict_next_word("Discussed preventative measures and healthy oral cleaning", 5)
# print([unique_tokens[idx] for idx in possible])

# generate a block of next
def generate_text(input_text, text_length, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current +=1
    return " ".join(word_sequence)

# generate x words and print to console
print(generate_text("Discussed preventative measures and healthy oral cleaning", 15, 5))