import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import transformers
import NERDA
import nltk

nltk.download('punkt')
from nltk import word_tokenize

data_1 = pd.read_csv('data_layer1.csv')
data_2 = pd.read_csv('data_layer2.csv')


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


data_layer1 = exclude_low_freq_classes(data_1, threshold=20)
data_layer2 = exclude_low_freq_classes(data_2, threshold=20)


def getuniquedomains(df, layer):
    unique_domains = df[layer].unique()
    unique_domains_dict = dict()
    for i, domain in enumerate(unique_domains):
        unique_domains_dict[domain] = i
    return unique_domains_dict


data_layer1_ddict = getuniquedomains(data_layer1, 'pseudoLabels')
data_layer2_ddict = getuniquedomains(data_layer2, 'pseudoLabels')

data_layer1['pseudoLabels'] = data_layer1['pseudoLabels'].map(lambda x: data_layer1_ddict[x])
data_layer2['pseudoLabels'] = data_layer2['pseudoLabels'].map(lambda x: data_layer2_ddict[x])

skill_split_layer1 = data_layer1['candidate_skill'].str.split(' ', expand=True)
skill_split_layer2 = data_layer2['candidate_skill'].str.split(' ', expand=True)
skill_split_layer1 = skill_split_layer1.add_prefix('token_')
skill_split_layer2 = skill_split_layer2.add_prefix('token_')

data_layer1 = skill_split_layer1.join(data_layer1)
data_layer2 = skill_split_layer2.join(data_layer2)

data_layer1['token_0'] = data_layer1['token_0'] + ':' + data_layer1['pseudoLabels'].astype(str)
data_layer1['token_1'] = data_layer1['token_1'] + ':' + data_layer1['pseudoLabels'].astype(str)
data_layer1['token_2'] = data_layer1['token_2'] + ':' + data_layer1['pseudoLabels'].astype(str)
data_layer1['token_3'] = data_layer1['token_3'] + ':' + data_layer1['pseudoLabels'].astype(str)

data_layer2['token_0'] = data_layer2['token_0'] + ':' + data_layer2['pseudoLabels'].astype(str)
data_layer2['token_1'] = data_layer2['token_1'] + ':' + data_layer2['pseudoLabels'].astype(str)
data_layer2['token_2'] = data_layer2['token_2'] + ':' + data_layer2['pseudoLabels'].astype(str)
data_layer2['token_3'] = data_layer2['token_3'] + ':' + data_layer2['pseudoLabels'].astype(str)

data_layer1 = data_layer1.fillna('')
data_layer2 = data_layer2.fillna('')

data_layer1['combined'] = data_layer1['left_context'] + ' ' + data_layer1['token_0'] + ' ' + data_layer1[
    'token_1'] + ' ' + data_layer1['token_2'] + ' ' + data_layer1['token_3'] + ' ' + data_layer1['right_context'] + '.'
data_layer2['combined'] = data_layer2['left_context'] + ' ' + data_layer2['token_0'] + ' ' + data_layer2[
    'token_1'] + ' ' + data_layer2['token_2'] + ' ' + data_layer2['token_3'] + ' ' + data_layer2['right_context'] + '.'

data_layer1_shuffled = shuffle(data_layer1, random_state=123)
data_layer2_shuffled = shuffle(data_layer2, random_state=123)

train_layer1 = data_layer1_shuffled[['combined']].iloc[0:round(0.8 * len(data_layer1_shuffled)), ]
val_layer1 = data_layer1_shuffled[['combined']].drop(train_layer1.index).iloc[
             0:round(0.1 * len(data_layer1_shuffled)), ]
test_layer1 = data_layer1_shuffled[['combined']].drop(train_layer1.index).drop(val_layer1.index).iloc[
              0:round(0.1 * len(data_layer1_shuffled)), ]

train_layer2 = data_layer2_shuffled[['combined']].iloc[0:round(0.8 * len(data_layer2_shuffled)), ]
val_layer2 = data_layer2_shuffled[['combined']].drop(train_layer2.index).iloc[
             0:round(0.1 * len(data_layer2_shuffled)), ]
test_layer2 = data_layer2_shuffled[['combined']].drop(train_layer2.index).drop(val_layer2.index).iloc[
              0:round(0.1 * len(data_layer2_shuffled)), ]

test_layer1 = data_layer1_shuffled[['combined']].sample(frac=0.2, random_state=123)
train_layer1 = data_layer1_shuffled.drop(test_layer1.index)
train_layer1 = train_layer1[['combined']]

test_layer2 = data_layer2_shuffled[['combined']].sample(frac=0.2, random_state=123)
train_layer2 = data_layer2_shuffled.drop(test_layer2.index)
train_layer2 = train_layer2[['combined']]


def tuple_transformer(dataset):
    token_df = pd.DataFrame()

    dataset = dataset

    for index in range(0, len(dataset['combined'])):
        tokens = pd.DataFrame(word_tokenize(dataset['combined'].iloc[index]), columns=['tokens'])
        tokens['index'] = index

        token_df = pd.concat([token_df, tokens])

    annotationcolumn = token_df['tokens'].str.split(':', expand=True)

    token_df['annotation'] = annotationcolumn[1]
    token_df['tokens'] = annotationcolumn[0]

    token_df['annotation'] = token_df['annotation'].fillna('0')

    token_df = token_df.rename(columns={'index': 'sentence'})

    token_df['tuples'] = list(zip(token_df.tokens, token_df.annotation))

    tuple_list = []
    sublist = []
    for row in token_df['tuples']:
        sublist.append(row)
        if row[0] == '.':
            tuple_list.append(sublist)
            sublist = []
            next
    # return sublist
    return tuple_list


training_layer1 = tuple_transformer(train_layer1)
validation_layer1 = tuple_transformer(val_layer1)
testing_layer1 = tuple_transformer(test_layer1)

training_layer2 = tuple_transformer(train_layer2)
validation_layer2 = tuple_transformer(val_layer2)
testing_layer2 = tuple_transformer(test_layer2)


def list_of_lists(dataset):
    texts = []
    tags = []

    token_sublist = []
    label_sublist = []

    for sublist in dataset:
        for token, label in sublist:

            token_sublist.append(token)
            label_sublist.append(label)
            if token == '.':
                texts.append(token_sublist)
                tags.append(label_sublist)

                token_sublist = []
                label_sublist = []
    return texts, tags


training_text_layer1, training_label_layer1 = list_of_lists(training_layer1)
val_text_layer1, val_label_layer1 = list_of_lists(validation_layer1)
test_text_layer1, test_label_layer1 = list_of_lists(testing_layer1)

training_text_layer2, training_label_layer2 = list_of_lists(training_layer2)
val_text_layer2, val_label_layer2 = list_of_lists(validation_layer2)
test_text_layer2, test_label_layer2 = list_of_lists(testing_layer2)


def dict_creator(text, label):
    from collections import defaultdict
    my_dict = defaultdict(list)
    for x in text:
        my_dict['sentences'].append(x)

    for x in label:
        my_dict['tags'].append(x)

    return dict(my_dict)


training_dict_layer1 = dict_creator(training_text_layer1, training_label_layer1)
validation_dict_layer1 = dict_creator(val_text_layer1, val_label_layer1)
testing_dict_layer1 = dict_creator(test_text_layer1, test_label_layer1)

training_dict_layer2 = dict_creator(training_text_layer2, training_label_layer2)
validation_dict_layer2 = dict_creator(val_text_layer2, val_label_layer2)
testing_dict_layer2 = dict_creator(test_text_layer2, test_label_layer2)

from NERDA.models import NERDA

# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
tag_scheme_layer1 = list(data_layer1_ddict.values())[1:]
tag_scheme_layer2 = list(data_layer2_ddict.values())[1:]
model_layer1 = NERDA(dataset_training=training_dict_layer1,
                     dataset_validation=validation_dict_layer1,
                     tag_scheme=tag_scheme_layer1,
                     tag_outside='0',
                     transformer='bert-base-uncased',
                     hyperparameters={'epochs': 10,
                                      'warmup_steps': 500,
                                      'train_batch_size': 13,
                                      'learning_rate': 0.0001}, )

model_layer2 = NERDA(dataset_training=training_dict_layer2,
                     dataset_validation=validation_dict_layer2,
                     tag_scheme=tag_scheme_layer2,
                     tag_outside='0',
                     transformer='bert-base-uncased',
                     hyperparameters={'epochs': 10,
                                      'warmup_steps': 500,
                                      'train_batch_size': 13,
                                      'learning_rate': 0.0001}, )
model_layer1_multi = NERDA(dataset_training=training_dict_layer1,
                           dataset_validation=validation_dict_layer1,
                           tag_scheme=tag_scheme_layer1,
                           tag_outside='0',
                           transformer='bert-base-multilingual-uncased',
                           hyperparameters={'epochs': 3,
                                            'warmup_steps': 500,
                                            'train_batch_size': 13,
                                            'learning_rate': 0.0001}, )

model_layer2_multi = NERDA(dataset_training=training_dict_layer2,
                           dataset_validation=validation_dict_layer2,
                           tag_scheme=tag_scheme_layer2,
                           tag_outside='0',
                           transformer='bert-base-multilingual-uncased',
                           hyperparameters={'epochs': 3,
                                            'warmup_steps': 500,
                                            'train_batch_size': 13,
                                            'learning_rate': 0.0001}, )

model_layer1.train()
model_layer1.save_network('Network_Layer1_deep')
model_layer2.train()
model_layer2.save_network('Network_Layer2_deep')
#model_layer1_multi.train()
#model_layer1_multi.save_network('Network_Layer1_Multi')
#model_layer2_multi.train()
#model_layer2_multi.save_network('Network_Layer2_Multi')

model_layer1.evaluate_performance(testing_dict_layer1)
model_layer2.evaluate_performance(testing_dict_layer2)
#model_layer1_multi.evaluate_performance(testing_dict_layer1)
#model_layer2_multi.evaluate_performance(testing_dict_layer2)
