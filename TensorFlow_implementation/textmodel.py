import string
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf

class SimpleVocab(object):

  def __init__(self):
    super(SimpleVocab, self).__init__()
    self.word2id = {}
    self.wordcount = {}
    self.word2id['<UNK>'] = 0
    self.wordcount['<UNK>'] = 9e9

  def tokenize_text(self, text):
    text = text
    tokens = str(text).lower().translate(str.maketrans('','',string.punctuation)).strip().split()
    return tokens

  def add_text_to_vocab(self, text):
    tokens = self.tokenize_text(text)
    
    for token in tokens:
      if token not in self.word2id:
        self.word2id[token] = len(self.word2id)
        self.wordcount[token] = 0
      self.wordcount[token] += 1

  def threshold_rare_words(self, wordcount_threshold=5):
    for w in self.word2id:
      if self.wordcount[w] < wordcount_threshold:
        self.word2id[w] = 0

  def encode_text(self, text):
    tokens = self.tokenize_text(text)
    x = [self.word2id.get(t, 0) for t in tokens]
    return x

  def get_size(self):
    return len(self.word2id)


class TextLSTMModel(tf.keras.Model):

  def __init__(self,
               texts_to_build_vocab,
               word_embed_dim,
               lstm_hidden_dim):
    super(TextLSTMModel, self).__init__()

    self.vocab = SimpleVocab()
    for text in texts_to_build_vocab:
      self.vocab.add_text_to_vocab(text)

    vocab_size = self.vocab.get_size()
    self.word_embed_dim = word_embed_dim
    self.lstm_hidden_dim = lstm_hidden_dim
    self.embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=word_embed_dim)
    self.gru=tf.keras.layers.LSTM(lstm_hidden_dim,return_sequences=False) 
    self.fc_output = tf.keras.Sequential([tf.keras.layers.Dropout(0.1), 
                                          tf.keras.layers.Dense(lstm_hidden_dim, input_shape=(lstm_hidden_dim,))])

  def call(self, x):

    x = [self.vocab.encode_text(text) for text in x]
    lengths = [len(t) for t in x]
    itexts = np.zeros((len(x),np.max(lengths)))
    for i in range(len(x)):
        itexts[ i,:lengths[i]]=(x[i])


    etexts = self.embedding_layer(itexts)
    # GRU
    lstm_output = self.forward_lstm_(etexts)
    text_features = self.fc_output(lstm_output)
    return text_features

  def forward_lstm_(self, etexts):
    batch_size = etexts.shape[1]
    lstm_output = self.gru(etexts)

    return lstm_output