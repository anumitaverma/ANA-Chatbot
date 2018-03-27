from keras import Sequential
import numpy as np
import pickle
import gensim
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.model_selection import train_test_split

import tensorflow as tf

with open('data.pickle') as f:
    vec_ques,vec_ans=pickle.load(f)

vec_ques=np.array(vec_ques,dtype=np.float64)
vec_ans=np.array(vec_ans,dtype=np.float64)

x_train,x_test,y_train,y_test=train_test_split(vec_ques,vec_ans,test_size=0.1)

model=Sequential()
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',
               inner_init='glorot_normal',activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=x_train.shape[1:],return_sequences=True,init='glorot_normal',
               inner_init='glorot_normal',activation='sigmoid'))
model.compile(loss='cosine_proxity',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,nb_epoch=250,validation_data=(x_test,y_test))
model.save('LSTM250.h5')
model.fit(x_train,y_train,nb_epoch=250,validation_data=(x_test,y_test))
model.save('LSTM500.h5')
model.fit(x_train,y_train,nb_epoch=250,validation_data=(x_test,y_test))
model.save('LSTM750.h5')
model.fit(x_train,y_train,nb_epoch=250,validation_data=(x_test,y_test))
model.save('LSTM1000.h5')
model.fit(x_train,y_train,nb_epoch=250,validation_data=(x_test,y_test))
model.save('LSTM1250.h5')

pred_out=model.predict(x_test)

#prediction for nothing


