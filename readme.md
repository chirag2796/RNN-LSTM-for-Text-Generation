# RNN LSTM Networks for Text Generation

This is a ML/Deep Learning and NLP project to train recurrent neural networks to generate Trump Tweets (text).


[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![Generic badge](https://img.shields.io/badge/tensorflow-1.14-orange.svg)](https://shields.io/)


## About the Project

*   A Recurent Neural Network with LSTM nodes implementation for text generation, trained on Donald Trump tweets dataset using contextual labels, and can generate realistic ones from random noise.
*   Ability to use Bidirectional RNNs, techniques such as attention-weighting and skip-embedding.
*   CuDNN implementation for training the RNNs on an nVidia GPU
*   This project has been specifically optimised for trump tweet dataset, and generating Trump tweets, but it can be used on any text dataset.

## Resources

*   [Live Demo](http://cgupta.tech/RnnTextGenerator.html) [![Website perso.crans.org](https://img.shields.io/website-up-down-green-red/http/perso.crans.org.svg)](http://perso.crans.org/)

*   [Trained Model Binaries](http://chirag2796.pythonanywhere.com/trump_tweet_model) [![Website perso.crans.org](https://img.shields.io/website-up-down-green-red/http/perso.crans.org.svg)](http://perso.crans.org/)

*   [Trump Tweet Dataset](http://chirag2796.pythonanywhere.com/trump_tweet_dataset) [![Website perso.crans.org](https://img.shields.io/website-up-down-green-red/http/perso.crans.org.svg)](http://perso.crans.org/)


## Implementation Guide



```bash
Clone this repository
Download the dataset (from Resources)
Download the dependencies, and CUDA
Train from train.py
```

## Usage

```python
# Simple Model Training (train.py)
from model.model import TextGenModel

model_config = {
    'name': 'trump_tweet_model',
    'meta_token': "<s>",
    'word_level': True,
    'rnn_layers': 2,
    'rnn_size': 512,
    'rnn_bidirectional': False,
    'max_length': 40,
    'max_words': 20000,
    'dim_embeddings': 100,
    'word_level': True,
    'single_text': False
}

trump_tweet_model = TextGenModel(model_config=model_config)
trump_tweet_model.train('trump_tweet_dataset.txt', header=False,
num_epochs=4, new_model=True)
```

## Architecture
![RNN Arvhitecture](http://cgupta.tech/images/rnn_representative.png)
*   The recurrent neural network takes sequence of words as input and outputs a matrix of probability for each word from dictionary to be the next of given sequence.
*   The model also learns how much similarity is between words or characters and calculates the probability of each.
*   Using that, it predicts or generates the next word or character of sequence.

![Bidirectional LSTM Network](http://cgupta.tech/images/bidirectional_lstm_small.jpg)
Representative image of model architecture for the Bidirectional LSTM network

## License
[MIT](https://choosealicense.com/licenses/mit/)
##
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/) [![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)