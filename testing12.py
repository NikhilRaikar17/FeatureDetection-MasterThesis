
import gensim.models

word2vec = './GoogleNews-vectors-negative300.bin'
word2vecModel = gensim.models.KeyedVectors.load_word2vec_format(word2vec,binary=True,limit=8000)
X = word2vecModel[word2vecModel.vocab]
# print(X)
print(word2vecModel.word_vec("man"))

print(word2vecModel.word_vec("woman"))


abc = (word2vecModel.word_vec("man") + word2vecModel.word_vec("woman")/(word2vecModel.word_vec("man") * word2vecModel.word_vec("woman")))

print(abc)

