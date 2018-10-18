import string
import collections
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import gensim

def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    text = text.translate(string.punctuation)
    tokens = word_tokenize(text)

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def cluster_texts(texts, clusters=5):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    # vectorizer = TfidfVectorizer(tokenizer=process_text,
    #                              stop_words=stopwords.words('english'),
    #                              max_df=0.5,
    #                              min_df=0.1,
    #                              lowercase=True)
    vectorizer = TfidfVectorizer(lowercase=True,analyzer = 'word', stop_words=stopwords.words('english'),tokenizer=process_text)
    tfidf_model = vectorizer.fit_transform(texts)

    print(tfidf_model.shape)

    km_model = KMeans(n_clusters=clusters,random_state=5)
    KMeans()
    km_model.fit(tfidf_model)
    X = tfidf_model.toarray()
    Agg_model = AgglomerativeClustering( n_clusters=clusters)
    Agg_model.fit(X)
    dbscan = DBSCAN(eps=9, min_samples=2).fit(X)

    clustering = collections.defaultdict(list)

    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
    #
    # for idx, label in enumerate(Agg_model.labels_):
    #     clustering[label].append(idx)
    #
    # for idx, label in enumerate(dbscan.labels_):
    #     clustering[label].append(idx)
    # for idx, topic in lda_model_tfidf.print_topics(-1):
    #     print('Topic: {} Word: {}'.format(idx, topic))

    return clustering


    # articles = ['Lack of time','Life is currently too busy to keep up with everything.  I would like to be more active in dealing with my Tinnitus.  More recently I had a hearing test and now have hearing aids, this has been a great help. I need to get back to my research and collaboration.','i cannot look in to it']
    clusters = cluster_texts(articles)
    pprint(dict(clusters))