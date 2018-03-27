import gensim
import csv
import nltk
import numpy as np
import pickle
from gensim import corpora, models,similarities

#nltk.download('punkt')

l=[]

word_to_vec= gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

#print(word_to_vec['Weegs'])
#l.append(word_to_vec['Weegs'])
#print(l)

data_set=[]

with open('output.csv','r',encoding='utf-8') as f:
    lines=csv.reader(f)


    for row in lines:
        data_set.append(row)
    f.close()


    ques=[]
    ans=[]

    for row in data_set:
        ques.append(row[0])
        ans.append(row[1])

    tok_ques=[]
    tok_ans=[]

    for i in range(len(ques)):
        tok_ques.append(nltk.word_tokenize(ques[i]))
        tok_ans.append(nltk.word_tokenize(ques[i]))


    print("Balle")
    vec_ques=[]
    vec_ans=[]


    for sublist in tok_ques:
        tempword = []
        for word in sublist:
            if(word in word_to_vec.vocab):
                tempword.append(word_to_vec[word])
        vec_ques.append(tempword)


    for sublist in tok_ans:
        tempword = []
        for word in sublist:
            if (word in word_to_vec.vocab):
                tempword.append(word_to_vec[word])
        vec_ans.append(tempword)

    """
    for row in vec_ans:
        print("====",row)"""

    print("Balle Balle")

    pad_vec = np.ones((300,), dtype=np.float32)


    ###change the implmenetation ----------


    for tok in vec_ques:
        tok=tok[:15]
        tok.append(pad_vec)

    for tok in vec_ques:
        if(len(tok)<15):
            for i in range(15-len(tok)):
                tok.append(pad_vec)

    print("Balle x3")


    ###change the implmenetation ----------
    for tok in vec_ans:
        tok=tok[:15]
        tok.append(pad_vec)

    for tok in vec_ans:
        if (len(tok) < 15):
            for i in range(15 - len(tok)):
                tok.append(pad_vec)

    print("finito")

    with open('data.pickle','w') as f:
        pickle.dump(str([vec_ques,vec_ans]),f)


