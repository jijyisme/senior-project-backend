import sys
import os
import pandas as pd
import csv

# Prevent Keras info message; "Using TensorFlow backend."
STDERR = sys.stderr
sys.stderr = open(os.devnull, "w")
sys.stderr = STDERR
sys.path.append(os.path.abspath('../'))
import numpy as np
# from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.models import load_model as load
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
np.set_printoptions(threshold=np.nan)

max_review_length = 1000

def read_all_files(path):
    collection = []
    for file in os.listdir(path):
        
        filename = os.fsdecode(file)

        if '.txt' in filename :
            temp = pd.read_csv(path+'/'+filename,sep='|',header=None, index_col=None, quoting=csv.QUOTE_NONE).values.tolist()
            collection.append(temp)
    return collection

def split_tag(collection):
    x_train = []
    pos_y_train = []
    ner_y_train = []
    for i in range(len(collection)):
        post = collection[i][0]
        splitted_word_list = []
        pos_list = []
        ner_list = []
        for j in range(len(post)):
            word = post[j]
            try:
                splitted_word, pos_tag, ner_tag = word.split('/')
                splitted_word_list.append(splitted_word)
                pos_list.append(pos_tag)
                ner_list.append(ner_tag)
            except:
                continue
        x_train.append(splitted_word_list)
        pos_y_train.append(pos_list)
        ner_y_train.append(ner_list)
    return x_train, pos_y_train, ner_y_train

def vectorize(raw_x_train):
    x_train = []
    preprocessor = Wordvectorizer()
    for post in raw_x_train:
        temp = preprocessor.predict([post], pre_process=False)
        x_train.append(temp)

    return x_train
def create_tag_list(collection):
    raw_x_train, raw_y_train = split_tag(collection)
    tag_list = np.unique(np.array(raw_y_train).flatten())
    print('lenlenlen',tag_list.shape)
    return tag_list

def decode_tag(y_dummy,tag_map):
    y_pred = [] 
    for post in y_dummy:
        temp = []
        for i in range(len(post)):
            temp.append(tag_map[np.argmax(y_dummy[0][i])])
        y_pred.append(temp)
    return y_pred

def encode_tag(padded_y_train, num_files, max_num_words):
    flatten_y_train = np.array(padded_y_train).reshape(num_files * max_num_words)    
    tag_list = np.unique(flatten_y_train)
    encoder = LabelEncoder()
    encoder.fit(np.array(tag_list))
    encoded_y_train = encoder.transform(flatten_y_train)
    dummy_y_train = np_utils.to_categorical(encoded_y_train).reshape(num_files,max_num_words,len(tag_list))
    return dummy_y_train, tag_list

def pad_tag(raw_y_train, num_files, max_num_words):
    padded_y_train = []
    for y in raw_y_train :
        padded_y_train.append((y+(['X']*max_num_words))[0:max_num_words])
    return padded_y_train

def prepare_train_data(collection, max_num_words):
    num_files = len(collection)
    raw_x_train, raw_pos_y_train, raw_ner_y_train = split_tag(collection)

    vector_x_train = vectorize(raw_x_train)

    padded_y_train = pad_tag(raw_ner_y_train, num_files, max_num_words)
    y_train, tag_list = encode_tag(padded_y_train, num_files, max_num_words)

    x_train = sequence.pad_sequences(vector_x_train, maxlen=max_num_words)

    return x_train, y_train, tag_list

def load_text_file(path):
    with open (path, "r") as file:
        text = file.readlines()[0]
    # text = codecs.open(path, 'r', 'utf8')
    print('input text\n',text)
    preprocessor = DataPreprocessor()
    tokenized_sentences, sentences_vector = preprocessor.preprocess_data(text)
    processed_text = sequence.pad_sequences([sentences_vector], maxlen=max_review_length)
    print('tokenized_sentences\n',tokenized_sentences)
    return processed_text, tokenized_sentences

def load_model(model_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model = load(dir_path+'/'+model_name)

    return model