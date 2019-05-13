# author - Richard Liao
# Dec 26 2016
import h5py
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup
import csv
import os
#import keras

os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.models import load_model
from keras.layers import add, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed

from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers


# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


MAX_SENT_LENGTH = 600
MAX_SENTS = 25
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def takeOne(elem):
    return elem[0]


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


data_train = pd.read_csv('fine.tsv', sep='\t')


print data_train.shape

from nltk import tokenize

reviews = []
labels = []
texts = []


for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx])
    text = clean_str(text.get_text().encode('ascii', 'ignore'))
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)

    labels.append(data_train.sentiment[idx])


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(texts)


data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')


for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1



word_index = tokenizer.word_index

print('Total %s unique tokens.' % len(word_index))


labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])


np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])


x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]

x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set')
print y_train.sum(axis=0)
print y_val.sum(axis=0)


# embedding

# GLOVE_DIR = "."
# embeddings_index = {}
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()

EMBEDDING_DIR = "./word_embedding"
embeddings_index = {}
f = open(os.path.join(EMBEDDING_DIR, 'fasttext_embedding.vec'))

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)
sentEncoder = Model(sentence_input, l_att)
#
review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_att_sent = AttLayer(100)(l_lstm_sent)
preds = Dense(10, activation='softmax')(l_att_sent)
model = Model(review_input, preds)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=50)


# save the weight maybe it can success
file = h5py.File("classfication_model_weight.h5", "w")
weight = model.get_weights()
for i in range(len(weight)):
    file.create_dataset('weight' + str(i), data=weight[i])
file.close()

'''
# save the weight maybe it can success
file = h5py.File("classification.h5", "w")
weight = model.get_weights()
for i in range(len(weight)):
    file.create_dataset('weight' + str(i), data=weight[i])
file.close()

'''
'''
file=h5py.File("./zhaobaiyang_model_weight", 'r')
weight = []
for i in range(len(file.keys())):
    weight.append(file['weight'+str(i)][:])
model.set_weights(weight)

'''
# For predict
for root, dirs, files in os.walk("/home/rjmonster/textClassifier/data/"):
    for file_i in files:

        predict_texts = []
        reviews_predict = []
        data_predict = pd.read_csv('/home/rjmonster/textClassifier/data/'+str(file_i), sep='\t')

        for idx in range(data_predict.review.shape[0]):
            predict_text = BeautifulSoup(data_predict.review[idx])
            predict_text = clean_str(predict_text.get_text().encode('ascii', 'ignore'))
            predict_texts.append(predict_text)
            sentences_predict = tokenize.sent_tokenize(predict_text)
            reviews_predict.append(sentences_predict)

        tokenizer_predict = Tokenizer(nb_words=MAX_NB_WORDS)
        tokenizer_predict.fit_on_texts(predict_texts)

        data_2 = np.zeros((len(predict_texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

        for i, sentences_predict in enumerate(reviews_predict):
            for j, sent in enumerate(sentences_predict):
                if j < MAX_SENTS:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(wordTokens):
                        if k < MAX_SENT_LENGTH and tokenizer_predict.word_index[word] < MAX_NB_WORDS:
                            data_2[i, j, k] = tokenizer_predict.word_index[word]
                            k = k + 1

        # word_index_predict = tokenizer_predict.word_index
        # print('Total %s unique predict tokens.' % len(word_index_predict))
        # indices_predict = np.arangde(data_2.shape[0])
        # np.random.shuffle(indices_predict)
        # data_2 = data_2[indices_predict]

        predict = model.predict(data_2, batch_size=50)
        # classes = predict.argmax(axis=-1)
        classes = np.argmax(predict, axis=1)

        result = []
        for index in range(0, len(reviews_predict)):
            paragraph = str(''.join(reviews_predict[index]))
            result.append((classes[index], paragraph))
        result.sort(key=takeOne)

        with open('./predict_result/'+str(file_i)+'.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(result)


