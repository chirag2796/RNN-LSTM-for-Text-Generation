import warnings
warnings.simplefilter(action='ignore')

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
trump_tweet_model.train('datasets\\trump_tweet_dataset.txt', header=False, num_epochs=4, new_model=True)
