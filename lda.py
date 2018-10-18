import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import string
import gensim
from gensim.corpora import Dictionary
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
import pyLDAvis.gensim

d_frame = pd.read_csv('data1.csv', encoding="ISO-8859-1")

print(d_frame.head())

# print(d_frame)
# df_filtered = d_frame['Other_new'].dropna()
doc_complete=d_frame.values.tolist()

flatten = [item for sublist in doc_complete for item in sublist]

print(flatten)

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in flatten]

print(doc_clean)


dct = Dictionary(doc_clean)

convert = pd.DataFrame.from_dict(doc_clean)

print(convert)

# print(dct)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dct.doc2bow(doc) for doc in doc_clean]

#Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=4, id2word = dct, passes=50)

print(ldamodel.print_topics(num_topics=4, num_words=20))
ldamodel.save('exp.gensim')

#Save the model
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

#Visualise LDA
lda = gensim.models.ldamodel.LdaModel.load('exp.gensim')
lda_display = pyLDAvis.gensim.prepare(lda,doc_term_matrix,dct)
pyLDAvis.show(lda_display)