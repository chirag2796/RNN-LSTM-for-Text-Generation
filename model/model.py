from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import multi_gpu_model
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
import json
import h5py
from pkg_resources import resource_filename
import csv
import re
from model.model_blueprint import init_rnn_model
from model import utils as Utils
from tqdm import trange

class TextGenModel:
    def __init__(self, model_config):
        self.config = model_config
        self.META_TOKEN = "<s>"

    def generate(self, n=1, return_as_list=False, prefix=None,
                 temperature=[1.0, 0.5, 0.2, 0.2],
                 max_gen_length=300, interactive=False,
                 top_n=3, progress=True):
        gen_texts = []
        iterable = trange(n) if progress and n > 1 else range(n)
        for _ in iterable:
            gen_text, _ = Utils.textgenrnn_generate(self.model,
                                              self.vocab,
                                              self.indices_char,
                                              temperature,
                                              self.config['max_length'],
                                              self.META_TOKEN,
                                              self.config['word_level'],
                                              self.config.get(
                                                  'single_text', False),
                                              max_gen_length,
                                              interactive,
                                              top_n,
                                              prefix)
            if not return_as_list:
                print("{}\n".format(gen_text))
            gen_texts.append(gen_text)
        if return_as_list:
            return gen_texts

    def generate_samples(self, n=5, temperatures=[0.2, 0.5, 1.0], **kwargs):
        for temperature in temperatures:
            print('#'*20 + '\nTemperature: {}\n'.format(temperature) +
                  '#'*20)
            self.generate(n, temperature=temperature, progress=False, **kwargs)

    @staticmethod
    def textgenrnn_texts_from_file(file_path, header=True, delim='\n'):
        '''
        Retrieves texts from a newline-delimited file and returns as a list.
        '''

        with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
            if header:
                f.readline()
            texts = [line.rstrip(delim) for line in f]

        return texts

    def train_new_model(self, texts, num_epochs=50,
                        gen_epochs=1, batch_size=16, dropout=0.5,
                        train_size=1.0,
                        validation=True, save_epochs=0, **kwargs):

        print("Training new model w/ {}-layer, {}-cell {}LSTMs".format(
            self.config['rnn_layers'], self.config['rnn_size'],
            'Bidirectional ' if self.config['rnn_bidirectional'] else ''
        ))

        # Create text vocabulary for new texts
        # if word-level, lowercase; if char-level, uppercase
        # self.tokenizer = Tokenizer(filters='', lower=True, char_level=False)
        self.tokenizer = Tokenizer(lower=True, char_level=False)
        self.tokenizer.fit_on_texts(texts)

        # Limit vocab to max_words
        max_words = self.config['max_words']
        self.tokenizer.word_index = {k: v for (k, v) in self.tokenizer.word_index.items() if v <= max_words}

        # if not self.config.get('single_text', False):
        #     self.tokenizer.word_index[self.META_TOKEN] = len(self.tokenizer.word_index) + 1
        self.vocab = self.tokenizer.word_index
        self.num_classes = len(self.vocab) + 1
        self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

        # Create a new, blank model w/ given params
        self.model = init_rnn_model(self.num_classes,
                                      dropout=dropout,
                                      cfg=self.config)

        # Save the files needed to recreate the model
        with open('{}_vocab.json'.format(self.config['name']),
                  'w', encoding='utf8') as outfile:
            json.dump(self.tokenizer.word_index, outfile, ensure_ascii=False)

        with open('{}_config.json'.format(self.config['name']),
                  'w', encoding='utf8') as outfile:
            json.dump(self.config, outfile, ensure_ascii=False)

        self.train_on_texts(texts, new_model=True,
                        num_epochs=num_epochs,
                        gen_epochs=gen_epochs,
                        train_size=train_size,
                        batch_size=batch_size,
                        dropout=dropout,
                        validation=validation,
                        save_epochs=save_epochs)

    def train_on_texts(self, texts, context_labels=None,
                       batch_size=16,
                       num_epochs=50,
                       verbose=1,
                       new_model=True,
                       gen_epochs=1,
                       train_size=1.0,
                       max_gen_length=75,
                       validation=True,
                       dropout=0.3,
                       save_epochs=0):


        if self.config['word_level']:
            # If training word level, must add spaces around each
            # punctuation. https://stackoverflow.com/a/3645946/9314418
            punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—…'
            for i in range(len(texts)):
                texts[i] = re.sub('([{}])'.format(punct), r' \1 ', texts[i])
                texts[i] = re.sub(' {2,}', ' ', texts[i])
            texts = [text_to_word_sequence(text, filters='') for text in texts]

        # calculate all combinations of text indices + token indices
        indices_list = [np.meshgrid(np.array(i), np.arange(len(text) + 1)) for i, text in enumerate(texts)]
        # indices_list = np.block(indices_list) # this hangs when indices_list is large enough
        # FIX BEGIN ------
        indices_list_o = np.block(indices_list[0])
        for i in range(len(indices_list)-1):
            tmp = np.block(indices_list[i+1])
            indices_list_o = np.concatenate([indices_list_o, tmp])
        indices_list = indices_list_o
        # FIX END ------

        # If a single text, there will be 2 extra indices, so remove them
        # Also remove first sequences which use padding
        # if self.config['single_text']:
        #     indices_list = indices_list[self.config['max_length']:-2, :]

        indices_mask = np.random.rand(indices_list.shape[0]) < train_size


        gen_val = None
        val_steps = None
        if train_size < 1.0 and validation:
            indices_list_val = indices_list[~indices_mask, :]
            gen_val = Utils.generate_sequences_from_texts(texts, indices_list_val, self, context_labels, batch_size)
            val_steps = max(int(np.floor(indices_list_val.shape[0] / batch_size)), 1)

        indices_list = indices_list[indices_mask, :]

        num_tokens = indices_list.shape[0]
        assert num_tokens >= batch_size, "Fewer tokens than batch_size."

        level = 'word' if self.config['word_level'] else 'character'
        print("Training on {:,} {} sequences.".format(num_tokens, level))

        steps_per_epoch = max(int(np.floor(num_tokens / batch_size)), 1)

        gen = Utils.generate_sequences_from_texts(texts, indices_list, self, context_labels, batch_size)

        base_lr = 4e-3

        # scheduler function must be defined inline.
        def lr_linear_decay(epoch):
            return (base_lr * (1 - (epoch / num_epochs)))

        if context_labels is not None:
            if new_model:
                weights_path = None
            else:
                weights_path = "{}_weights.hdf5".format(self.config['name'])
                self.save(weights_path)

            self.model = init_rnn_model(self.num_classes,
                                          dropout=dropout,
                                          cfg=self.config,
                                          context_size=context_labels.shape[1],
                                          weights_path=weights_path)

        model_t = self.model


        model_t.fit_generator(gen, steps_per_epoch=steps_per_epoch,
                              epochs=num_epochs,
                              callbacks=[
                                  LearningRateScheduler(lr_linear_decay),
                                  Utils.generate_after_epoch(self, gen_epochs, max_gen_length),
                                  Utils.save_model_weights(self, num_epochs, save_epochs)
                              ],
                              verbose=verbose,
                              max_queue_size=10,
                              validation_data=gen_val,
                              validation_steps=val_steps)

        # Keep the text-only version of the model if using context labels
        if context_labels is not None:
            self.model = Model(inputs=self.model.input[0],
                               outputs=self.model.output[1])



    def train(self, file_path, header=True, delim="\n", **kwargs):
            texts = TextGenModel.textgenrnn_texts_from_file(file_path, header, delim)
            # texts = texts[:100]
            print("{:,} texts collected.".format(len(texts)))
            self.train_new_model(texts, **kwargs)
