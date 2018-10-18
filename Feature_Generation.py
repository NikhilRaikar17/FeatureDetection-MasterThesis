import nltk
from nltk.chunk import conlltags2tree, tree2conlltags
nltk.download('maxent_ne_chunker')
nltk.download('words')

def named_Entity(cluster):

    text = nltk.word_tokenize(cluster)
    ne = nltk.ne_chunk(nltk.pos_tag(text))
    iob_tagged = tree2conlltags(ne)

    NNJJ = []
    nnp = []
    noun_add = []

    #To retrieve Root nodes using NNP tags
    for j in range(0, len(iob_tagged)-1):
        if(iob_tagged[j][1] == 'NNP'):
            noun = iob_tagged[j]
            nnp .append(noun)

    #To retrieve adjectives and Nouns in the sentences
    for i in range(0, len(iob_tagged) - 1):
        if (iob_tagged[i][1] == 'JJ' and iob_tagged[i + 1][1] == 'NN'):
            w = iob_tagged[i] + iob_tagged[i+1]
            NNJJ.append(w)

        if(iob_tagged[i][1] == 'NN' and iob_tagged[i + 1][1] == 'NN'):
            noun = iob_tagged[i] + iob_tagged[i + 1]
            noun_add.append(noun)


    fre_nnp = nltk.FreqDist(nnp)
    for word, frequency in fre_nnp.most_common(15):
        if(frequency >= 1):
            print(u'{};{}'.format(word, frequency))
            break

    fre_noun = nltk.FreqDist(noun_add)
    for word, frequency in fre_noun.most_common(15):
        print(u'{};{}'.format(word, frequency))

    fre_nnjj = nltk.FreqDist(NNJJ)
    for word, frequency in fre_nnjj.most_common(15):
        if (frequency >= 2):
            print(u'{};{}'.format(word, frequency))



