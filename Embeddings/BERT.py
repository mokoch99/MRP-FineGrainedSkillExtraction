
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

import pickle
import json

# Sklearn
from sklearn.model_selection import train_test_split # for splitting data into train and test samples
from sklearn.svm import SVC # for Support Vector Classification baseline model
from sklearn.semi_supervised import SelfTrainingClassifier # for Semi-Supervised learning
from sklearn.metrics import classification_report # for model evaluation metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification




df = pd.read_csv(r'Final_Data_Semisupervised_Learning.csv')



df = df[df["sentiment"].notnull()]

df['left_context'] = df['left_context'].fillna('empty')
df['right_context'] = df['right_context'].fillna('empty')

df['left_context'] = df['left_context'].apply(lambda x: x.strip())
df['right_context'] = df['right_context'].apply(lambda x: x.strip())
df['candidate_skill'] = df['candidate_skill'].apply(lambda x: x.strip())



df['concatenated'] = df['left_context'] + ' | ' + df['candidate_skill'] + ' | ' + df['right_context']


embedding = TransformerWordEmbeddings("Ivo/emscad-skill-extraction")

count = 0
def bert_embedder(text):

    global count
    count += 1
    if(count%1000 == 0):
      print(count)

    string = Sentence(text)
    embedding.embed(string)

    # Creating a list which stores the indexes of the | symbols
    bar_indexes = []


    #Checking the sentence object for the | symbols and storing their indexes
    for x in range(1,len(string)+1):
        if '|' in str(string.get_token(x)):
            bar_indexes.append(x)

    #Creating a list which stores the embedding_tensors
    embedding_tensors = 0
    #Collecting the embeddings for every index between the indexes in bar_indexes
    word_embedding_indexes = range(bar_indexes[0]+1,bar_indexes[1])
    for x in word_embedding_indexes:
        embedding_tensors += pd.Series(string[x].embedding)
        #embedding_tensors.append(5)

    #Turning the elements from embedding_tensors into dataframe rows
    row = pd.DataFrame()
    for x in range(0,len(embedding_tensors)):
        row = row.append(pd.DataFrame(pd.Series(embedding_tensors[x])))

    row = row.transpose().reset_index(drop=True)

    #Changing the column names in order to make pd.concat work later
    row.columns = [x for x in range(0,len(row.columns))]

    return row



df = df[df["sentiment"] == '-']


df.reset_index(drop = True,inplace = True)

df['embeddings'] = df['concatenated'].apply(lambda x: bert_embedder(x))



df["Vector_Len"] = np.nan
for i in range(len(df)):
  df["Vector_Len"][i] = df['embeddings'][i].shape[1]


df.to_pickle("embeddings_data.pkl")
