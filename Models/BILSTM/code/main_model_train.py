import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from imblearn.over_sampling import SMOTE

def exclude_low_freq_classes(data, threshold=20):
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
    file = '95% pseudolabels w2v - layer2 - cleaned.csv' # 95% pseudolabels w2v - layer1 - cleaned.csv # 95% pseudolabels - fasttext - layer1 - cleaned.csv # 95% pseudolabels - bert embeddings - layer1 - cleaned.csv
    embeddings = 'custom fasttext' # custom w2v # custom fasttext # custom bert
    layer = 'layer 2' # layer 2
    contextual = 'True' # False
    oversampling = 'False' # True
    classifier = 'Bidirectional LSTM'

    data = pd.read_csv('/Users/kiliankramer/Desktop/experiments/' + file)
    data = exclude_low_freq_classes(data)

    # create X
    X = data.iloc[:, 8:-1]
    X = X.values.tolist()
    '''
    for i, sample in enumerate(X):
        emb1 = sample[:100]
        emb2 = sample[100:200]
        emb3 = sample[200:]
        X[i] = [emb1, emb2, emb3]
    '''
    # create Y
    output_classes = data.pseudoLabels.nunique()
    unique_domains = data.pseudoLabels.unique()
    domain_dict = dict()
    domain_dict_reverse = dict()
    for i, domain in enumerate(unique_domains):
        domain_dict[domain] = i
        domain_dict_reverse[i] = domain
    Y = data.pseudoLabels.replace(domain_dict)
    Y = Y.tolist()

    print(unique_domains)
    print(output_classes)
    print(domain_dict)
    print(domain_dict_reverse)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    # If selected, perform oversampling to the training data
    if oversampling == 'True':
        smote = SMOTE(random_state = 42)
        x_train, y_train = smote.fit_resample(x_train, y_train)

    if contextual == 'True':
        for i, sample in enumerate(x_train):
            emb1 = sample[:100]
            emb2 = sample[100:200]
            emb3 = sample[200:]
            x_train[i] = [emb1, emb2, emb3]
        for i, sample in enumerate(x_test):
            emb1 = sample[:100]
            emb2 = sample[100:200]
            emb3 = sample[200:]
            x_test[i] = [emb1, emb2, emb3]
    elif contextual == 'False':
        for i, sample in enumerate(x_train):
            emb2 = sample[100:200]
            x_train[i] = [emb2]
        for i, sample in enumerate(x_test):
            emb2 = sample[100:200]
            x_test[i] = [emb2]

    # MODEL PART
    model = Sequential([
        LSTM(100),
        Dense(output_classes, activation='relu'),
        Dense(output_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    num_epochs = 100

    log_folder = 'logs'
    callbacks = [
        EarlyStopping(patience=10),
        TensorBoard(log_dir=log_folder)
    ]

    history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=callbacks)

    # LOAD MODEL
    # model = keras.models.load_model('/Users/kiliankramer/Desktop/models/LOL')

    # SAVEl MODEL
    contextual_word = 'contextual_false'
    oversampling_word = 'oversampling_false'
    if contextual == 'True':
        contextual_word = 'contextual_true'
    if oversampling == 'True':
        oversampling_word = 'oversampling_true'
    model.save("/Users/kiliankramer/Desktop/experiments/models/" + embeddings + " " + layer + " " + contextual_word + " " + oversampling_word)

    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test accuracy :', accuracy)

    # Predict the test set labels
    y_pred = model.predict(x_test)
    y_pred_list = []
    for i in y_pred:
        y_pred_list.append(np.argmax(i, axis=0))
    y_pred = y_pred_list

    # Back Mapping
    y_test_pd = pd.DataFrame(y_test, columns=['col1'])
    y_pred_pd = pd.DataFrame(y_pred, columns=['col1'])
    y_test = y_test_pd.col1.replace(domain_dict_reverse)
    y_pred = y_pred_pd.col1.replace(domain_dict_reverse)

    # Report results
    print("-----------------------------------------")
    print(f"Dataset : {file}")
    print(f"Embedding : {embeddings}")
    print(f"Labels : {layer}")
    print(f"Contextual : {contextual}")
    print(f"Oversampling : {oversampling}")
    print(f"Classifier : {classifier}")
    print(f"Number of training instances : {len(y_train)}")
    print(classification_report(y_test, y_pred))
    # print(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred)}")
    # print(f"Average Precision : {average_precision_score(y_test, y_pred)}")

    if os.path.exists("/Users/kiliankramer/Desktop/experiments/results/" + embeddings + " " + layer + " " + contextual_word + " " + oversampling_word + ".txt"):
        os.remove("/Users/kiliankramer/Desktop/experiments/results/" + embeddings + " " + layer + " " + contextual_word + " " + oversampling_word + ".txt")
    with open("/Users/kiliankramer/Desktop/experiments/results/" + embeddings + " " + layer + " " + contextual_word + " " + oversampling_word + ".txt", 'w') as f:
        f.write(f"Dataset : {file}\n")
        f.write(f"Embedding : {embeddings}\n")
        f.write(f"Labels : {layer}\n")
        f.write(f"Contextual : {contextual}\n")
        f.write(f"Oversampling : {oversampling}\n")
        f.write(f"Classifier : {classifier}\n")
        f.write(f"Number of training instances : {len(y_train)}\n")
        f.write(classification_report(y_test, y_pred))







