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

if __name__ == '__main__':
    # Load Pre-trained word embeddings, and sentences and outuput classes from training dataset
    data = pd.read_csv('/Users/kiliankramer/Desktop/final_data_fine_grained.csv')
    embeddings = gensim.models.Word2Vec.load("/Users/kiliankramer/Desktop/model1")

    x = []
    y = []

    embeddings_index = {}

    # Some preprocessing

    for index, row in data.iterrows():
        sentence_origin = str(row['left_context']) + str(" ") + str(row['candidate_skill']) + str(" ") + str(row['right_context'])
        tokens = sentence_origin.split()
        sentence_new = ""
        for token in tokens:
            if token in embeddings.wv:
                embeddings_index[token] = embeddings.wv[token]
                sentence_new += str(" ") + str(token)
        x.append(sentence_new[1:])
        y.append(row['layer1'])

    y = pd.Series((v for v in y))

    somedict = {}
    c = 0
    for row in y:
        if row not in somedict:
            somedict[row] = c
            c += 1
    for key in somedict:
        y = y.replace(key, somedict[key])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    vocab_size = 10000
    oov_token = "<OOV>"
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(x_train)

    word_index = tokenizer.word_index

    x_train_sequences = tokenizer.texts_to_sequences(x_train)
    x_test_sequences = tokenizer.texts_to_sequences(x_test)

    max_length = 100
    padding_type = 'post'
    truncation_type = 'post'

    x_test_padded = pad_sequences(x_test_sequences, maxlen=100,
                                  padding=padding_type, truncating=truncation_type)
    x_train_padded = pad_sequences(x_train_sequences, maxlen=100, padding=padding_type,
                                   truncating=truncation_type)

    '''
    embeddings_index = {}
    f = open('/Users/kiliankramer/Desktop/glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    '''

    # Create embedding matrix for embedding layer

    embedding_matrix = np.zeros((len(word_index) + 1, max_length))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                output_dim=100,
                                weights=[embedding_matrix],
                                input_length=100,
                                trainable=False)

    # What to adapt here?

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(150, return_sequences=True)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model = keras.models.load_model('/Users/kiliankramer/Desktop/BiLSTM')

    log_folder = 'logs'
    callbacks = [
        EarlyStopping(patience=10),
        TensorBoard(log_dir=log_folder)
    ]
    num_epochs = 600
    history = model.fit(x_train_padded, y_train, epochs=num_epochs, validation_data=(x_test_padded, y_test),
                        callbacks=callbacks)

    loss, accuracy = model.evaluate(x_test_padded, y_test)
    print('Test accuracy :', accuracy)

    # model.save("/Users/kiliankramer/Desktop/BiLSTM")


    # print("Hi")