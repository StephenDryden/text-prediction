from model import model, n_words, unique_tokens, unique_token_index
import numpy as np
import random
from flask import Flask, request, jsonify

# Load the trained model and other necessary variables
app = Flask(__name__)

def predict_next_word_api(input_text, n_best):
    input_text = input_text.lower()
    x = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        if word in unique_token_index:
            x[0, i, unique_token_index[word]] = 1
        else:
            # Handle unknown words
            pass

    predictions = model.predict(x)[0]
    possible_indices = np.argpartition(predictions, -n_best)[-n_best:]
    predicted_words = [unique_tokens[idx] for idx in possible_indices]
    return predicted_words

def generate_text_api(input_text, text_length, creativity=3):
    word_sequence = input_text.lower().split()
    for _ in range(text_length):
        sub_sequence = " ".join(word_sequence[-n_words:])
        possible_indices = predict_next_word_api(sub_sequence, creativity)
        choice = random.choice(possible_indices)
        word_sequence.append(choice)
    return " ".join(word_sequence)

@app.route('/predict_next_word', methods=['POST'])
def predict_next_word_endpoint():
    input_text = request.json.get('input_text')
    n_best = request.json.get('n_best')
    predicted_words = predict_next_word_api(input_text, n_best)
    return jsonify({'predicted_words': predicted_words})

@app.route('/generate_text', methods=['POST'])
def generate_text_endpoint():
    input_text = request.json.get('input_text')
    text_length = request.json.get('text_length')
    creativity = request.json.get('creativity', 3)
    generated_text = generate_text_api(input_text, text_length, creativity)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run()
