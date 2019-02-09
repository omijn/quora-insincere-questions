import datetime
import json
import operator
import os
import re
import string

import gensim
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.layers import CuDNNLSTM, Embedding, Dense, Dropout, Bidirectional, BatchNormalization, \
    Conv1D, MaxPooling1D, Flatten, LeakyReLU, SpatialDropout1D, Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.optimizers import Adam

dt = datetime.datetime.now().isoformat()
kaggle_mode = False
read_preprocessed_data = False

if kaggle_mode:
    train_path = "../input/train.csv"
    test_path = "../input/test.csv"
    embedding_paths = {
        'word2vec': "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin",
        'glove': "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"}
else:
    train_path = 'train.csv'
    test_path = 'train.csv'
    embedding_paths = {'word2vec': "/data/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin",
                       'glove': "/data/glove.840B.300d/glove.840B.300d.txt"}

preprocessed_train_file = 'train_preprocessed.csv'
savepath = "saves/{}".format(dt)
checkpoint_path = "{}/checkpoint".format(savepath)
tokenizer_path = "{}/tokenizer".format(savepath)
model_path = "{}/model".format(savepath)
history_path = "{}/history".format(savepath)
tensorboard_path = "{}/tensorboard/".format(savepath)

use_embedding = True
embedding_type = "word2vec"
embedding_path = embedding_paths[embedding_type]

if use_embedding:
    if embedding_type == "word2vec":
        embedding = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    elif embedding_type == "glove":
        gensim.scripts.glove2word2vec.glove2word2vec(embedding_path, embedding_path + ".word2vecformat")
        embedding = gensim.models.KeyedVectors.load_word2vec_format(embedding_path + ".word2vecformat")

    all_puncts = string.punctuation.replace("'", "")
    puncts_in_embedding = r"".join([punct for punct in all_puncts if punct in embedding])
    puncts_not_in_embedding = r"".join(set(all_puncts) - set(puncts_in_embedding))

    punctuation_to_keep = re.compile(r"([" + puncts_in_embedding + r"])")
    remove_punctuation = True
    try:
        punctuation_to_remove = re.compile(r"[" + re.escape(puncts_not_in_embedding) + r"]")
    except:
        remove_punctuation = False

    weird_punctuation = re.compile(r"’“”")
    numbers = re.compile(r"\b\d+")


def build_vocab(sentences):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def check_coverage(vocab, embeddings_index):
    intersection = {}  # words in vocab and embedding
    oov = {}  # words in vocab but not in embedding
    num_known_words = 0
    num_unknown_words = 0
    for word in vocab:
        try:
            intersection[word] = embeddings_index[word]
            num_known_words += vocab[word]
        except:
            oov[word] = vocab[word]
            num_unknown_words += vocab[word]

    print('Found embeddings for {:.2%} of vocab'.format(len(intersection) / len(vocab)))
    print('Found embeddings for {:.2%} of all text'.format(num_known_words / (num_known_words + num_unknown_words)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


spelling_transformations = {
    "behaviour": "behavior",
    "offence": "offense",
    "neighbour": "neighbor",
    "colour": "color",
    "travelling": "traveling",
    "organisation": "organization",
    "masterbate": "masturbate",
    "grey": "gray",
    "tyre": "tire"
}


def transform_question_text(question_text):
    # for words not in embedding, remove ' appearing at the beginning/end of word: countries', 'white
    # to replace contractions not in embedding, try the following in this order: lowercase, removing 's, lowercase
    # to replace other words not in embedding, try to lemmatize

    question_text = question_text.replace(r'”', '"')
    question_text = question_text.replace(r'“', '"')
    question_text = question_text.replace(r"’", "'")
    question_text = punctuation_to_keep.sub(r' \1 ', question_text)
    if remove_punctuation:
        question_text = punctuation_to_remove.sub(' ', question_text)
    question_text = numbers.sub(lambda match: "#" * len(match.group(0)), question_text)

    words_to_remove = ['to', 'and', 'the', 'of']

    cleaned_words = []
    for word in question_text.split():
        if word in embedding:
            cleaned_words.append(word)
        elif word in words_to_remove:
            pass
        else:
            if word.startswith("'"):
                word_to_check = word[1:]
                if word_to_check in embedding:
                    cleaned_words.append(word_to_check)
                elif word_to_check.lower() in embedding:
                    cleaned_words.append(word_to_check.lower())
                else:
                    cleaned_words.append(word)
            elif word.endswith("'"):
                word_to_check = word[:-1]
                if word_to_check in embedding:
                    cleaned_words.append(word_to_check)
                elif word_to_check.lower() in embedding:
                    cleaned_words.append(word_to_check.lower())
                else:
                    cleaned_words.append(word)
            elif "'" in word:
                if word.lower() in embedding:
                    cleaned_words.append(word.lower())
                    break

                word_to_check = re.sub(r"'.*", "", word)
                if word_to_check in embedding:
                    cleaned_words.append(word_to_check)
                elif word_to_check.lower() in embedding:
                    cleaned_words.append(word_to_check.lower())
                else:
                    cleaned_words.append(word)
            else:
                for s in spelling_transformations:
                    if s in word:
                        word = word.replace(s, spelling_transformations[s])
                cleaned_words.append(word)

    return " ".join(cleaned_words)


def create_pretrained_embedding_layer(tokenizer, input_length):
    num_embeddings = len(tokenizer.word_index) + 1
    embedding_dimension = len(embedding['orange'])
    embedding_matrix = np.random.random((num_embeddings, embedding_dimension))

    for word, index in tokenizer.word_index.items():
        try:
            embedding_matrix[index, :] = embedding[word]
        except KeyError:
            continue

    embedding_layer = Embedding(input_dim=num_embeddings, output_dim=embedding_dimension,
                                input_length=input_length, weights=[embedding_matrix], trainable=False)

    return embedding_layer


class F1(Callback):

    def on_train_begin(self, logs=None):
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.asarray(self.model.predict(self.validation_data[0])).round()
        y_true = self.validation_data[1]
        f1 = f1_score(y_true, y_pred)
        self.f1_scores.append(f1)
        print(" - val_f1: {:.4f}".format(f1))


# read data
data = pd.read_csv(train_path)


def model1():
    X = data['question_text']
    y = data['target']
    np.random.seed(42)
    Xtrain, Xval, ytrain, yval = train_test_split(X, y, train_size=0.994)

    tfidf = TfidfVectorizer(max_features=80000, stop_words=['to', 'and', 'the', 'of', 'in', 'a'], ngram_range=(1, 3))
    tfidf.fit(Xtrain)

    Xtrain = tfidf.transform(Xtrain)
    Xval = tfidf.transform(Xval)

    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xval)
    print(f1_score(yval, ypred))


# model1()

X = data['question_text']
y = data['target']
X = X.apply(lambda question_text: transform_question_text(question_text))

if kaggle_mode:
    Xtrain = X
    ytrain = y
    test_data = pd.read_csv(test_path)
    Xval = test_data['question_text']
    Xval = Xval.apply(lambda question_text: transform_question_text(question_text))

else:
    np.random.seed(42)
    Xtrain, Xval, ytrain, yval = train_test_split(X, y, train_size=0.994)

# Xtrain = Xtrain[:50000]
# ytrain = ytrain[:50000]
# Xval = Xval[:5000]
# yval = yval[:5000]

tokenizer = Tokenizer(filters='', lower=False)
tokenizer.fit_on_texts(Xtrain)

Xtrain = tokenizer.texts_to_sequences(Xtrain)
Xval = tokenizer.texts_to_sequences(Xval)

QUESTION_SIZE_LIMIT = 25
max_question_size = len(max(Xtrain, key=len))
if max_question_size > QUESTION_SIZE_LIMIT:
    max_question_size = QUESTION_SIZE_LIMIT

Xtrain = pad_sequences(Xtrain, max_question_size)
Xval = pad_sequences(Xval, max_question_size)

f1_callback = F1()
if not kaggle_mode:
    os.makedirs(savepath, exist_ok=True)

# create model
model = Sequential()
model.add(create_pretrained_embedding_layer(tokenizer, max_question_size))
# model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(72, return_sequences=False)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.002), loss='binary_crossentropy')

model2 = Sequential()
model2.add(create_pretrained_embedding_layer(tokenizer, max_question_size))
model2.add(Conv1D(64, 4, activation='relu'))
model2.add(MaxPooling1D(2))
# model2.add(Conv1D(64, 4, activation='relu'))
# model2.add(MaxPooling1D(2))
model2.add(Flatten())
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')

earlystopping = EarlyStopping(patience=3, verbose=True)
checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False)
reduce_lr = ReduceLROnPlateau(verbose=True, patience=3)
tensorboard = TensorBoard(log_dir=tensorboard_path, write_graph=False, write_grads=True, batch_size=128,
                          histogram_freq=1)

if kaggle_mode:
    callbacks = None
    val_data = None
    verbose = False
else:
    callbacks = [checkpoint, f1_callback, earlystopping]
    val_data = (Xval, yval)
    verbose = True

# history = model.fit(Xtrain, ytrain, epochs=50, batch_size=256, validation_data=val_data,
#                     callbacks=callbacks, class_weight={0: 0.25, 1: 0.75}, verbose=verbose)

history = model.fit(Xtrain, ytrain, epochs=50, batch_size=256, validation_data=val_data, callbacks=callbacks,
                    class_weight={0: 0.25, 1: 0.75}, verbose=verbose)

if kaggle_mode:
    y_pred = model.predict_classes(Xval)
    submission = pd.DataFrame({'qid': test_data.qid, 'prediction': y_pred.flatten()})
    submission.to_csv('submission.csv', index=False)

else:
    joblib.dump(tokenizer, tokenizer_path)
    model.save(model_path)
    with open(history_path, "w") as f:
        f.write(json.dumps(history.history))
        f.write("\n\n")
        f.write(json.dumps(history.params))
        f.write("\n\n")
        f.write("F1 scores: " + json.dumps(f1_callback.f1_scores))
        f.write("\n")
