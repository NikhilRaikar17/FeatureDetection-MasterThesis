from gensim.test.utils import common_texts, get_tmpfile

from gensim.models import Word2Vec

import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('german.model', binary=True)

print(model.vocab.keys())

