import nltk
from nltk import word_tokenize
import string
import textrank_sentences
from nltk.stem import WordNetLemmatizer
import preprocessing
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gensim
from sklearn.cluster import KMeans
import collections
import  Feature_Generation
encoding="utf-8"
nltk.download('punkt')

sentences = preprocessing.words_reviews



Text = textrank_sentences.Text

def clean(text):
    text = text.lower()
    printable = set(string.printable)
    #text = filter(lambda x: x in printable, text) #filter funny characters, if any.
    return text

Cleaned_text = clean(Text)
text = word_tokenize(Cleaned_text)

POS_tag = nltk.pos_tag(text)

wordnet_lemmatizer = WordNetLemmatizer()

adjective_tags = ['JJ', 'JJR', 'JJS']

lemmatized_text = []

for word in POS_tag:
    if word[1] in adjective_tags:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0], pos="a")))
    else:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0])))  # default POS = noun

POS_tag = nltk.pos_tag(lemmatized_text)

stopwords = []

wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW']

for word in POS_tag:
    if word[1] not in wanted_POS:
        stopwords.append(word[0])

punctuations = list(str(string.punctuation))

stopwords = stopwords + punctuations

stopword_file = open("long.txt", "r")
#Source = https://www.ranks.nl/stopwords

lots_of_stopwords = []

for line in stopword_file.readlines():
    lots_of_stopwords.append(str(line.strip()))

stopwords_plus = []
# stopwords_plus = stopwords + lots_of_stopwords
stopwords_plus = lots_of_stopwords
stopwords_plus = set(stopwords_plus)

processed_text = []
for word in lemmatized_text:
    if word not in stopwords_plus:
        processed_text.append(word)

vocabulary = list(set(processed_text))

import numpy as np
import math

vocab_len = len(vocabulary)

weighted_edge = np.zeros((vocab_len, vocab_len), dtype=np.float32)

score = np.zeros((vocab_len), dtype=np.float32)
window_size = 3
covered_coocurrences = []

for i in range(0, vocab_len):
    score[i] = 1
    for j in range(0, vocab_len):
        if j == i:
            weighted_edge[i][j] = 0
        else:
            for window_start in range(0, (len(processed_text) - window_size)):

                window_end = window_start + window_size

                window = processed_text[window_start:window_end]

                if (vocabulary[i] in window) and (vocabulary[j] in window):

                    index_of_i = window_start + window.index(vocabulary[i])
                    index_of_j = window_start + window.index(vocabulary[j])

                    # index_of_x is the absolute position of the xth term in the window
                    # (counting from 0)
                    # in the processed_text

                    if [index_of_i, index_of_j] not in covered_coocurrences:
                        weighted_edge[i][j] += 1 / math.fabs(index_of_i - index_of_j)
                        covered_coocurrences.append([index_of_i, index_of_j])

inout = np.zeros((vocab_len),dtype=np.float32)

for i in range(0,vocab_len):
    for j in range(0,vocab_len):
        inout[i]+=weighted_edge[i][j]


MAX_ITERATIONS = 50
d = 0.85
threshold = 0.0001  # convergence threshold

for it in range(0, MAX_ITERATIONS):
    prev_score = np.copy(score)

    for i in range(0, vocab_len):

        summation = 0
        for j in range(0, vocab_len):
            if weighted_edge[i][j] != 0:
                summation += (weighted_edge[i][j] / inout[j]) * score[j]

        score[i] = (1 - d) + d * (summation)

    if np.sum(np.fabs(prev_score - score)) <= threshold:  # convergence condition
        break

for i in range(0,vocab_len):
    print("Score of "+vocabulary[i]+": "+str(score[i]))


example_dictionary= dict(zip(vocabulary, score))

print(example_dictionary)


word2vec = './GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec,binary=True,limit=5000)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
X = model[model.vocab]
vocab = model.vocab.keys()

# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(w2v))])
        else:
            self.dim = 0

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * (example_dictionary.get(w) if example_dictionary.get(w) is not None else 0)
                     for w in words if w in self.word2vec and example_dictionary.get(w) is not None] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec[next(iter(w2v))])

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

# a= MeanEmbeddingVectorizer(w2v)
a=TfidfEmbeddingVectorizer(w2v)
a.fit(sentences)
b= a.transform(sentences)

km_model = KMeans(n_clusters=20,random_state=5)
km_model.fit(b)

clustering = collections.defaultdict(list)

for idx, label in enumerate(km_model.labels_):
    clustering[label].append(idx)

a=dict(clustering.items())
# for i,j in zip(a.values(),a.keys()):
clus=[]
for value,key in zip(a.values(),a.keys()):
    print('----------------cluster' + str(key)+' ' + '-------------')
    for words in value:
        clus.append(" ".join(sentences[words]))
    Feature_Generation.named_Entity(""" """.join(clus))
    clus = []
    print('----------------cluster' + str(key) + ' ' + '-------------')