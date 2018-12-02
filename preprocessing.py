import io
from itertools import chain
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re

ab = list(chain(*[[word_tokenize(sent) for sent in sent_tokenize(line)] for line in io.open('waste.txt', 'r', encoding='utf8')]))

stop_words = set(stopwords.words('english'))

words_reviews = []
for i in ab:
    wr=[]
    for word in i:
        if word not in stop_words:
            wr.append(word)
    words_reviews.append(wr)

print(words_reviews)
