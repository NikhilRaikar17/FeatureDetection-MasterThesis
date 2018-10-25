from gensim.models import Word2Vec
import gensim.models
import numpy as np
from nltk.cluster import KMeansClusterer
import nltk
from sklearn import cluster
from numpy.linalg import norm
from sklearn import metrics
from pprint import pprint
import preprocessing
import collections

sentences = preprocessing.words_reviews

# sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
#              ['this', 'is', 'another', 'book'],
#              ['one', 'more', 'book'],
#              ['this', 'is', 'the', 'new', 'post'],
#              ['this', 'is', 'about', 'machine', 'learning', 'post'],
#              ['orange', 'juice', 'is', 'the', 'liquid', 'extract', 'of', 'the', 'fruit'],
#              ['orange', 'juice', 'comes', 'in', 'several', 'different', 'varieties'],
#              ['and', 'this', 'is', 'the', 'last', 'post']]



word2vec = './GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec,binary=True,limit=5000)
X = model[model.vocab]
vocab = model.vocab.keys()
wordsInVocab = len(vocab)

# print (model.similarity('post', 'book'))

def sent_vectorizer(sent, model):
    sent_vec = np.zeros(50)
    numw = 0
    for w in sent:
        try:
            vc = model[w]
            vc = vc[0:50]
            sent_vec = np.add(sent_vec, vc)
            numw += 1
        except:
            pass
    return sent_vec / np.sqrt(sent_vec.dot(sent_vec))

V = []
for sentence in sentences:
    V.append(sent_vectorizer(sentence, model))


results = [[0 for i in range(len(V))] for j in range(len(V))]

# for i in range(len(V) - 1):
#     for j in range(i + 1, len(V)):
#         NVI = norm(V[i])
#         NVJ = norm(V[j])
#         dotVij = 0
#         NVI = 0
#         for x in range(50):
#             NVI = NVI + V[i][x] * V[i][x]
#         NVJ = 0
#         for x in range(50):
#             NVJ = NVJ + V[j][x] * V[j][x]
#
#         for x in range(50):
#             dotVij = dotVij + V[i][x] * V[j][x]
#
#         results[i][j] = dotVij / (NVI * NVJ)
# print(results)

NUM_CLUSTERS = 10
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25,avoid_empty_clusters = True)
assigned_clusters = kclusterer.cluster(V, assign_clusters=True)
print(assigned_clusters)

words = list(model.vocab)

# for i, word in enumerate(words):
#     print(word + ":" + str(assigned_clusters[i]))

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS,random_state=5)
kmeans.fit(V)

labels = kmeans.labels_


centroids = kmeans.cluster_centers_
print(centroids)

for index, sentence in enumerate(sentences):
    print (str(assigned_clusters[index]) + ":" + str(sentence))

print("Cluster id labels for inputted data")
print(labels)

#
print("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print(kmeans.score(V))
#
clustering = collections.defaultdict(list)
for idx, j in enumerate(labels):
    clustering[j].append(idx)


print(clustering.items())










