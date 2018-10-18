import string
import collections
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
nltk.download('stopwords')
nltk.download('punkt')


def process_text(text, stem=True):
    text = text.translate(string.punctuation)
    tokens = word_tokenize(text)

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def cluster_texts(texts, clusters=5):

    vectorizer = TfidfVectorizer(lowercase=True,analyzer = 'word', stop_words=stopwords.words('english'),tokenizer=process_text)
    tfidf_model = vectorizer.fit_transform(texts)

    km_model = KMeans(n_clusters=clusters,random_state=5)
    KMeans()
    km_model.fit(tfidf_model)

    clustering = collections.defaultdict(list)

    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)

    return clustering


    clusters = cluster_texts(var = art)
    pprint(dict(clusters))