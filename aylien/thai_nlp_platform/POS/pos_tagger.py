import csv
import gc
import glob
import json
import os
import shutil
import sys
import warnings
from collections import Counter
from datetime import datetime
from multiprocessing import Process, Queue
from pprint import pprint

# Prevent Keras info message; "Using TensorFlow backend."
STDERR = sys.stderr
sys.stderr = open(os.devnull, "w")
from keras.models import load_model
sys.stderr = STDERR

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

from thai_nlp_platform.POS import constant
from thai_nlp_platform.POS.model import Model
import sklearn.metrics
import pandas as pd

class POSTagger(object):

    def __init__(self, model_path=None, new_model=False):

        self.new_model = new_model
        self.model_path = model_path
        self.model = None

        if not self.new_model:
            if model_path is not None:
                self.model = load_model(model_path)
            else:
                self.model = load_model(constant.DEFULT_MODEL_PATH)
            
    def evaluate(self, x_true, y_true, tag_index):
        if self.new_model:
            print("model is not trained")
            return

        model = self.model
        
        tag_index = index_builder(constant.TAG_LIST, constant.TAG_START_INDEX)

        pred = model.predict(x_true)
        amax_pred = np.argmax(pred, axis=2)
        amax_true = np.argmax(y_true, axis=2)
        print('acc: ',sklearn.metrics.accuracy_score(amax_true.flatten(),amax_pred.flatten()))
        print('f1 micro: ',sklearn.metrics.f1_score(y_pred=amax_pred.flatten(),y_true=amax_true.flatten(),average='micro'))
        print('f1 macro: ',sklearn.metrics.f1_score(y_pred=amax_pred.flatten(),y_true=amax_true.flatten(),average='macro'))
        result = (amax_pred.flatten() == amax_true.flatten())
        p=amax_pred.flatten()
        t = amax_true.flatten()
        r = result
        result_table = pd.DataFrame(data = {
                            'predict' : p,
                            'true' : t,
                            'result' : r
                        }
                    )
        most_incorrect_prediction_lable = result_table[result_table['result']==False]['predict'].value_counts()
        count_label = result_table['predict'].value_counts()
        print('++++++++++++++++++++++detail+++++++++++++++++++++')
        for index in most_incorrect_prediction_lable.index:
            print(index,'\t',
                most_incorrect_prediction_lable[index]/count_label[index],'\t',
                most_incorrect_prediction_lable[index],'\t',
                count_label[index],'\t',
                constant.TAG_LIST[index-2],'\t')


    def train(self, x_true, y_true, model_path=None, num_step=60, valid_split=0.1,
              initial_epoch=None, epochs=100, batch_size=32, learning_rate=0.001,
              shuffle=False):
        """Train model"""

        # Create new model or load model

        if self.new_model:
            initial_epoch = 0
            model = Model(num_step).model

        else:
            if not model_path:
                raise Exception("Model path is not defined.")

            if initial_epoch is None:
                raise Exception("Initial epoch is not defined.")

            model = load_model(model_path)

        # Display model summary before train
        model.summary()
        
        # Train model
        model.fit(x_true, y_true, validation_split=valid_split,
                  initial_epoch=initial_epoch, epochs=epochs,
                  batch_size=batch_size, shuffle=shuffle)
        self.model = model
        self.new_model = False

    def save(self, path, name):
        # Save model architecture to file
        with open(os.path.join(path, name+".json"), "w") as file:
            file.write(self.model.to_json())

        # Save model config to file
        with open(os.path.join(path, name+"_config.txt"), "w") as file:
            pprint(self.model.get_config(), stream=file)

        self.model.save(os.path.join(path,name+'.hdf5'))

    def predict(self, x_vector):
        if self.new_model:
            print("model is not trained")
            return
        model = self.model

        per = model.predict(x_vector)
        amax = np.argmax(per, axis=2)
        predict_y = amax.flatten()
        x_flat = x_vector.flatten()
        
        return dict({
            'x': x_flat,
            'pos_tag': predict_y
        })
