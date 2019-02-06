import os
from scipy import spatial
import numpy as np
import gensim
import nltk
from keras.models import load_model

model = load_model("LSTM_chatbot.h5")

pretrained_embeddings_path = "GoogleNews-vectors-negative300.bin"

model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)

while(True):
    x=input("Enter the message:");
    
    if(x=="bye"):
        break
    
    sentend=np.ones((300,),dtype=np.float32) 

    sent=nltk.word_tokenize(x.lower())
    sentvec = [model_word2vec[w] for w in sent if w in model_word2vec.wv.vocab]

    sentvec[19:]=[]
    sentvec.append(sentend)
    
    if len(sentvec)<20:
        for i in range(20-len(sentvec)):
            sentvec.append(sentend) 
            
    sentvec=np.array([sentvec])
    
    predictions = model.predict(sentvec)
    outputlist=[model_word2vec.most_similar([predictions[0][i]])[0][0] for i in range(20)]
    output=' '.join(outputlist)
    print (output)