import preprocessing
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gensim
from sklearn.cluster import KMeans
import collections
import  Feature_Generation
sentences = preprocessing.words_reviews
encoding="utf-8"

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
        print(len(np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])))
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


a=TfidfEmbeddingVectorizer(w2v)
a.fit(sentences)
b= a.transform(sentences)

km_model = KMeans(n_clusters=10,random_state=5)
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
    print(clus)
    clus = []
    print('----------------cluster' + str(key) + ' ' + '-------------')
































# etree_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)),
#                         ("extra trees", ExtraTreesClassifier(n_estimators=200))])
#
#
#
#
# all_models = [
#     ("mult_nb", mult_nb),
#     ("mult_nb_tfidf", mult_nb_tfidf),
#     ("bern_nb", bern_nb),
#     ("bern_nb_tfidf", bern_nb_tfidf),
#     ("svc", svc),
#     ("svc_tfidf", svc_tfidf),
#     ("w2v", etree_w2v),
#     ("w2v_tfidf", etree_w2v_tfidf),
#     ("glove_small", etree_glove_small),
#     ("glove_small_tfidf", etree_glove_small_tfidf),
#     ("glove_big", etree_glove_big),
#     ("glove_big_tfidf", etree_glove_big_tfidf),
#
# ]
#
#
# unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
# scores = sorted(unsorted_scores, key=lambda x: -x[1])
#
#
# print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
#
#
# sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])
#
#
# #
# # model = Pipeline([
# #     ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
# #     ("extra trees", ExtraTreesClassifier(n_estimators=200))])
# #
# # etree_w2v_tfidf = Pipeline([
# #     ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
# #     ("extra trees", ExtraTreesClassifier(n_estimators=200))])
# #
# # X = [['Berlin', 'London'],
# #      ['cow', 'cat'],
# #      ['pink', 'yellow']]
# # y = ['capitals', 'animals', 'colors']
# #
# # model.fit(X, y)
#
# # # never before seen words!!!
# # test_X = [['dog'], ['red'], ['Madrid']]
# #
# # print(model.predict(test_X))
# # # prints ['animals' 'colors' 'capitals']