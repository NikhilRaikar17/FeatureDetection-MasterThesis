import string
import collections
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import pyclust

#Import statements for clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from spherecluster import SphericalKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn import datasets, cluster
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

#Import statements for vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import kmedoids

from pprint import pprint
nltk.download('stopwords')
nltk.download('punkt')
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

def process_text(text, stem=True):
    text = text.translate(string.punctuation)
    tokens = word_tokenize(text)

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def cluster_texts(texts, clusters):

    # Tf-Idf Vectorizer
    vectorizer = TfidfVectorizer(lowercase=True,analyzer = 'word', stop_words=stopwords.words('english'),tokenizer=process_text)

    #Count Vectorizer
    # vectorizer = CountVectorizer(lowercase=True, analyzer='word', stop_words=stopwords.words('english'),tokenizer=process_text)

    # Hash Vectorizer
    # vectorizer = HashingVectorizer(lowercase=True, analyzer='word', stop_words=stopwords.words('english'),tokenizer=process_text)

    # Hash Vectorizer
    #vectorizer = HashingVectorizer(lowercase=True, analyzer='word', stop_words=stopwords.words('english'),tokenizer=process_text)


    tfidf_model = vectorizer.fit_transform(texts)

    # print(tfidf_model.shape)
    # print(tfidf_model)


    #Kmeans Clustering
    km_model = KMeans(n_clusters=clusters,init = 'k-means++',random_state=5)
    km_model.fit(tfidf_model)

    ## Performing KMedoids
    X = tfidf_model.toarray()
    kmd = pyclust.KMedoids(n_clusters=10, n_trials=11)
    kmd.fit(X)



    # M, C = kmedoids.kMedoids(tfidf_model, 20)
    #
    # print('medoids:')
    # for point_idx in M:
    #     print(tfidf_model[point_idx])
    #
    # print('')
    # print('clustering result:')
    # for label in C:
    #     for point_idx in C[label]:
    #         print('label {0}:　{1}'.format(label, tfidf_model[point_idx]))

    #Adjust_Rand_Score
    # print("Top terms per cluster:")
    # order_centroids = km_model.cluster_centers_.argsort()[:, ::-1]
    # terms = vectorizer.get_feature_names()
    # for i in range(clusters):
    #     print("Cluster %d:" % i)
    #     for ind in order_centroids[i, :10]:
    #         print(' %s' % terms[ind])

    # Sil_score to select clusters Kmeans
    # for k in range(2, 20):
    #     # define k-means constructor
    #     kmeans = KMeans(n_clusters=k, random_state=10)
    #
    #     cluster_labels = kmeans.fit_predict(tfidf_model.toarray())
    #
    #     # Calculating silhouette_score for k
    #
    #     score = silhouette_score(tfidf_model.toarray(), cluster_labels, random_state=10)
    #     print("The silhouette score for {} clusters is {}".format(k, score))

    #Agglomerative Clustering
    # X = tfidf_model.toarray()
    # Agg_model = AgglomerativeClustering(n_clusters=clusters)
    # Agg_model.fit(X)

    #Spherical-Kmeans
    # spherical_kmeans = SphericalKMeans(n_clusters=clusters)
    # spherical_kmeans.fit(tfidf_model)

    #Affinity Propogation
    # AffinityProp = AffinityPropagation(affinity='euclidean').fit(tfidf_model)

    #Feature Agglomerative
    # tfidf = np.array(tfidf_model.todense())
    # featureagglo = cluster.FeatureAgglomeration(n_clusters=clusters)
    # featureagglo.fit(tfidf)

    #DBSCAN algorithm
    # X = tfidf_model.toarray()
    # dbscan = DBSCAN(eps=10, min_samples=1, algorithm='ball_tree', metric='minkowski', leaf_size=90, p=2).fit(X)
    # print("Clusters assigned are:", set(dbscan.labels_))




    clustering = collections.defaultdict(list)

    for idx, label in enumerate(kmd.labels_):
        clustering[label].append(idx)

    return clustering


    clusters = cluster_texts(var = art)
    pprint(dict(clusters))

