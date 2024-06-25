# Text-Prediction
A simple example of text prediction using dentistry notes

## How to run the basic example
```shell
source venv/bin/activate
python3 main.py
```
This will give you an example of 5 predicted words based on the sentance "Discussed preventative measures and healthy oral cleaning".
It will also generate some sample text based on the sentance "Oral cleaning completed. We discussed preventative"

## How to deploy the flask api

> NOTE: The model rebuilds each time the api is started and I'm not sure why. It should just load the already built model

```shell
source venv/bin/activate
python3 api.py
```

### predict_next_word endpoint
Command
```shell
curl -X POST -H "Content-Type: application/json" -d '{"input_text": "Oral cleaning completed. We discussed next", "n_best": 5}' http://127.0.0.1:5000/predict_next_word
```
Response
```json
{"predicted_words":["techniques","affecting","denture","around","after"]}
```

### generate_text endpoint
Command
```shell
curl -X POST -H "Content-Type: application/json" -d '{"input_text": "The quick brown fox", "text_length": 10, "creativity": 4}' http://127.0.0.1:5000/generate_text
```
Response
```json
{"generated_text":"the quick brown fox use comfort instructed instructed on different flossing with whitening and"}
```

## Considerations
* Data - We could do with better quality data and more of it
* Number of characters - if we get more data we can read in more characters = better model
* Number of epochs - more epochs = better accuracy but longer to train
* Number of words required to predict - not fully sure the impact of this, possibly more starting words = higher accuracy
* Number of words to return

## Key Explanations

### Categorical Cross-Entropy Loss :
This is a loss function commonly used for multi-class classification problems, where the goal is to assign an input sample to one of several mutually exclusive classes.

In this code, the model is trying to predict the next word in a sequence, which can be thought of as a multi-class classification problem, where each class is a unique word in the vocabulary.

The categorical cross-entropy loss measures the performance of the model by comparing the predicted probability distribution over the classes (words) with the true distribution (the one-hot encoded target word). The loss is calculated as the negative log-likelihood of the true class, summed over all examples.

By minimizing this loss during training, the model learns to assign higher probabilities to the correct words in the sequence.

### RMSprop (Root Mean Square Propagation) :
RMSprop is an optimization algorithm used to update the weights of the neural network during training. It is an adaptive learning rate method that helps the model converge faster and avoid getting stuck in local minima.

The key idea behind RMSprop is to divide the learning rate for a weight by a running average of the recent magnitudes of the gradients for that weight. This way, the learning rate is adjusted adaptively for each weight based on the historical gradients.

RMSprop is often used as an alternative to other popular optimizers like Stochastic Gradient Descent (SGD) or Adam, as it can converge faster and handle sparse gradients better in certain situations.

In the code, the model is compiled with the categorical cross-entropy loss function and the RMSprop optimizer with a learning rate of 0.01:

```python
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
```

This means that during training, the model will try to minimize the categorical cross-entropy loss by updating the weights using the RMSprop algorithm with a learning rate of 0.01.