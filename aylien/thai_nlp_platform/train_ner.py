#!/usr/bin/env python
import numpy as np
from utils.loader import read_all_files, prepare_train_data, create_tag_list,decode_tag, split_tag, pad_tag

from ner.model import NameEntityRecognizer
from ner.evaluation import evaluate

#use-case-01 : load existing model from the model_path
#Warning : x_train and y_train must match the input shape requirement

# max_num_words = 500
# vector_size = 100

# collection = read_all_files("ner/BEST_mock")
# x, y, ner_tag_mapping = prepare_train_data(collection, max_num_words)
# ner = NameEntityRecognizer(model_path='ner/models/test.h5')
# ner.train(x,y)
# y_pred = ner.predict(x)
# result = decode_tag(y_pred, ner_tag_mapping)






#use-case-02 : evaluate the model

collection = read_all_files("ner/BEST_mock")
x_train, pos_y_train, ner_y_train = split_tag(collection)
padded_y_train = pad_tag(ner_y_train, len(collection), 500)
x, y, ner_tag_mapping = prepare_train_data(collection, max_num_words = 500)

ner = NameEntityRecognizer(max_num_words = 500, word_vec_length=100)
ner.add_lstm_layer(input_shape=(500,100),bidirectional=True)
ner.add_lstm_layer(input_shape=(500,100),bidirectional=True)
ner.add_dense_layer(output_length=len(ner_tag_mapping))
ner.compile()

ner.train(x,y)
y_pred = ner.predict(x)
y_pred = decode_tag(y_pred, ner_tag_mapping)

scores = evaluate(padded_y_train ,y_pred)

ner.save('ner/models/test.h5')

for i in scores:
	print(i,scores[i])





