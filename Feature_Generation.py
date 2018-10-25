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
    trigr = []
    NN = []

    #To retrieve Root nodes using NNP tags
    for j in range(0, len(iob_tagged)-1):
        if(iob_tagged[j][1] == 'NNP'):
            noun = iob_tagged[j]
            nnp .append(noun)

        if (iob_tagged[j][1] == 'NN'):
            noun1 = iob_tagged[j]
            NN.append(noun1)


    #To retrieve adjectives and Nouns in the sentences
    for i in range(0, len(iob_tagged) - 1):
        if (iob_tagged[i][1] == 'JJ' and iob_tagged[i + 1][1] == 'NN'):
            w = iob_tagged[i] + iob_tagged[i+1]
            NNJJ.append(w)

        if(iob_tagged[i][1] == 'NN' and iob_tagged[i + 1][1] == 'NN'):
            nnnn = iob_tagged[i] + iob_tagged[i + 1]
            noun_add.append(nnnn)

        # if(iob_tagged[i][1] == 'JJ' and iob_tagged[i + 1][1] == 'NN' and iob_tagged[i + 2][1] == 'NN' ):
        #     trigram = iob_tagged[i] + iob_tagged[i + 1] + iob_tagged[i + 2]
        #     trigr.append(trigram)



    fre_nnp = nltk.FreqDist(nnp)
    for word, frequency in fre_nnp.most_common(15):
        print(u'{};{}'.format(word, frequency))

    fre_noun = nltk.FreqDist(noun_add)
    for word, frequency in fre_noun.most_common(15):
        print(u'{};{}'.format(word, frequency))

    fre_nnjj = nltk.FreqDist(NNJJ)
    for word, frequency in fre_nnjj.most_common(15):
        print(u'{};{}'.format(word, frequency))

    fre_trinn = nltk.FreqDist(trigr)
    for word, frequency in fre_trinn.most_common(15):
        print(u'{};{}'.format(word, frequency))

    fre_onlynn = nltk.FreqDist(NN)
    for word, frequency in fre_onlynn.most_common(15):
        print(u'{};{}'.format(word, frequency))



    # ne_tree = conlltags2tree(iob_tagged)
    # print(ne_tree)



