import gensim
import csv
import nltk
import numpy as np
from gensim import corpora, models,similarities

#nltk.download('punkt')



word_to_vec= gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

for work in word_to_vec.vocab:
    print("==========")
    print(work)


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



    vec_ques=[]
    vec_ans=[]
    for sublist in tok_ques:
        for word in sublist:
            if(word in word_to_vec.vocab):
                tempword=word_to_vec[word]
                vec_ques.append(tempword)

    for sublist in tok_ans:
        for word in sublist:
            if (word in word_to_vec.vocab):
                tempword = word_to_vec[word]
                vec_ans.append(tempword)


    """for row in tok_ans:
        print(row)"""

    default_vec = np.ones((300,), dtype=np.float32)

    for tok in tok_ques:
        tok[14:]=[]
        tok.append(default_vec)
        print("---",tok)


