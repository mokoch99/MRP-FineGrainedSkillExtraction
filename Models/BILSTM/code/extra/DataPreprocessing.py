import pandas as pd
import gensim
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import csv
import os

def exclude_low_freq_classes(data, threshold=10):
    df = data.copy()

    freq_table = df["pseudoLabels"].value_counts()
    low_freq_classes = []
    for i in range(len(freq_table)):
        if freq_table[i] < threshold:
            low_freq_classes.append(freq_table.index[i])

    for i in range(len(low_freq_classes)):
        df = df[df["pseudoLabels"] != low_freq_classes[i]]

    df.reset_index(drop=True, inplace=True)

    return df

if __name__ == '__main__':
    folder = '/Users/kiliankramer/Desktop/experiments'
    file = '95% pseudolabels - fasttext - layer1 - cleaned.csv'
    # 95% pseudolabels w2v - layer1 - cleaned.csv

    data = pd.read_csv(folder + '/' + file)
    # data = data[:100]
    data = exclude_low_freq_classes(data)
    X = data.iloc[:, 108:208]
    print(X.head)

    # create x
    embeddings = []
    for index, row in data.iterrows():
        print(index)
        new_embedding1 = []
        new_embedding2 = []
        new_embedding3 = []
        for i in range(0, 100):
            new_embedding1.append(data.iloc[index][str(i)])
            new_embedding2.append(data.iloc[index][str(i) + str(".1")])
            new_embedding3.append(data.iloc[index][str(i) + str(".2")])
        embeddings.append([new_embedding1, new_embedding2, new_embedding3])

    # create y
    unique_domains = data.pseudoLabels.unique()
    domain_dict = dict()
    for i, domain in enumerate(unique_domains):
        domain_dict[domain] = i
    print(domain_dict)
    y = data.pseudoLabels.replace(domain_dict)

    print(data["pseudoLabels"].nunique())
    print(data["pseudoLabels"].unique())

    # write x
    fields = ['left_context', 'skill_candidate', 'right_context']
    if os.path.exists(folder + '/processed_datasets/x ' + file):
        os.remove(folder + '/processed_datasets/x ' + file)
    with open(folder + '/processed_datasets/x ' + file, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(embeddings)
    # write y
    y.to_csv(folder + '/processed_datasets/y ' + file, index=False)
