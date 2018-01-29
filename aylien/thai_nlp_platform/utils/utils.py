"""
Utilities Function
"""

import glob
import math
import os
import re
import string
import gensim
import deepcut
import sys

import numpy as np
from keras.utils.np_utils import to_categorical

# sys.path.append('./utils/')
# import co
from thai_nlp_platform.utils import co



class Text(object):
    def __init__(self, path, filename, content):
        self.path = path
        self.filename = filename
        self.content = content
        self.token_list = None
    
    def __str__(self):
        return """
        path: {}
        filename: {}
        content: {}
        """.format(self.path,self.filename,self.content)


class TextCollection(object):
    """
    Corpus Manager

    This class provide files loading mechanism and keep files in memory to optimize read performance

    Args:
        corpus_directory (str): relative path to corpus directory
        text_mode (bool): is files contain ground truth 
        word_delimeter (str): the character that separate each words 
        tag_delimeter (str): the character that separate word and tags 

    limitation :
        this object is purpose for BEST 2010 corpus. Changing tag delimeter or word delimeter may cause of error. 
    """

    def __init__(self, corpus_directory=None, tokenize_function = None, word_delimiter='|', tag_delimiter='/', tag_dictionary = {'word': 0,'pos': 1,'ner': 2} ):
        # Global variable

        self.corpus_directory = corpus_directory
        self.word_delimiter = word_delimiter
        self.tag_delimiter = tag_delimiter
        self.tokenize_function = tokenize_function
        self.tag_dictionary = tag_dictionary
        self.__corpus = list()

        # Load corpus to memory
        self._load()

    def _load(self):
        """Load text to memory"""

        corpus_directory = glob.escape(self.corpus_directory)
        file_list = sorted(glob.glob(os.path.join(corpus_directory, "*.txt")))

        for path in file_list:
            with open(path, "r", encoding="utf8") as text:
                # Read content from text file
                content = text.read()

                # Preprocessing
                content = self._preprocessing(content)

                # Create text instance
                text = Text(path, os.path.basename(path), content)

                # Add text to corpus
                self.__corpus.append(text)

    def _preprocessing(self, content):
        """Text preprocessing"""

        # Remove new line
        content = re.sub(r"(\r\n|\r|\n)+", r"", content)

        # Convert one or multiple non-breaking space to space
        content = re.sub(r"(\xa0)+", r"\s", content)

        # Convert multiple spaces to only one space
        content = re.sub(r"\s{2,}", r"\s", content)

        # Trim whitespace from starting and ending of text
        content = content.strip(string.whitespace)

        if self.word_delimiter and self.tag_delimiter:
            # Trim word delimiter from starting and ending of text
            content = content.strip(self.word_delimiter)

            # Convert special characters (word and tag delimiter)
            # in text's content to escape character
            find = "{0}{0}{1}".format(re.escape(self.word_delimiter),
                                      re.escape(self.tag_delimiter))
            replace = "{0}{2}{1}".format(re.escape(self.word_delimiter),
                                         re.escape(self.tag_delimiter),
                                         re.escape(co.ESCAPE_WORD_DELIMITER))
            content = re.sub(find, replace, content)

            find = "{0}{0}".format(re.escape(self.tag_delimiter))
            replace = "{1}{0}".format(re.escape(self.tag_delimiter),
                                      re.escape(co.ESCAPE_TAG_DELIMITER))
            content = re.sub(find, replace, content)

        # Replace distinct quotation mark into standard quotation
        content = re.sub(r"\u2018|\u2019", r"\'", content)
        content = re.sub(r"\u201c|\u201d", r"\"", content)

        return content

    def add_text(self, content, path="", filename = ""):
        # Preprocessing
        content = self._preprocessing(content)

        # Create text instance
        text = Text(path, filename, content)

        # Add text to corpus
        self.__corpus.append(text)

    @property
    def count(self):
        return len(self.__corpus)

    def get_filename(self, index):
        return self.__corpus[index].filename

    def get_content(self, index):
        return self.__corpus[index].content

    def get_text(self, index):
        return self.__corpus[index]

    def get_token_list(self, index):

        if self.tokenize_function is not None:
            return self.tokenize_function(self.get_content(index))
        else:
            if self.__corpus[index].token_list is not None:
                return self.__corpus[index].token_list
            elif self.tag_delimiter is not None and self.word_delimiter is not None:
                self.__corpus[index].token_list = []
                content = self.get_content(index)

                token_list = content.split(self.word_delimiter)

                for idx, token in enumerate(token_list):
                    # Empty or Spacebar
                    if token == "" or token == co.SPACEBAR:
                        word = co.SPACEBAR
                        pos_tag = co.PAD_TAG_INDEX
                        ner_tag = co.PAD_TAG_INDEX

                    # Word
                    else:
                        # Split word and tag by tag delimiter
                        datum = token.split(self.tag_delimiter)

                    # Replace token with word and tag pair
                    self.__corpus[index].token_list.append(datum)

                return self.__corpus[index].token_list
            else:
                raise Exception('tag or word delimeter is missing and No tokenize is given')





class VectorCollection(object):
    """"""
    def __init__(self, x, y, readable_x,  readable_y):
        self.x=x
        self.y=y
        self.readable_x=readable_x
        self.readable_y=readable_y
    


def build_input(corpus, tag_index, num_step, vectorize_function = None,needed_y='pos'):
    if corpus.tokenize_function is None :
        x, y, readable_x, readable_y = generate_x_y(corpus=corpus, tag_index=tag_index,  vectorize_function = vectorize_function, needed_y = needed_y)
        
        x = pad(x,num_step,[0]*co.DATA_DIM)
        y = pad(y,num_step,0)
        readable_x = pad(readable_x,num_step,'<NULL>')
        readable_y = pad(readable_y,num_step,'<NULL>')

        return VectorCollection(x=np.array(x),y=np.array(y),readable_x=readable_x,readable_y=readable_y)
    else:
        x, readable_x = generate_x(corpus, vectorize_function)
        return VectorCollection(x=x, readable_x=readable_x)

def pad(x, num_step, pad_with):
    for i in range(len(x)):
        if len(x[i]) < num_step:
            x[i].extend([pad_with]*(num_step-len(x[i])))
        else :
            x[i] = x[i][0:num_step]
    return x


def generate_x_y(corpus, tag_index, vectorize_function,  needed_y):
    
    x = []
    readable_x = []
    y = []
    readable_y = []

    for corpus_idx in range(corpus.count):

        fx = []
        freadable_x = []
        fy = []
        freadable_y = []

        token_list = corpus.get_token_list(corpus_idx)
        # print(token_list)
        
        for token in token_list:
        
            index_x = corpus.tag_dictionary['word']
            word = token[index_x]
            fx.append(vectorize_function(word))
            freadable_x.append(word)
            if word == ' ' or word == '' :
                fy.append(tag_index[' '])
                freadable_y.append('<space>')
            else :
                index = corpus.tag_dictionary[needed_y]
                freadable_y.append(token[index])
                fy.append(tag_index[token[index]])
        x.append(fx)
        y.append(fy)
        readable_x.append(freadable_x)
        readable_y.append(freadable_y)
    return x, y, readable_x, readable_y

def generate_x(corpus, vectorize_function):
    x = []
    readable_x = []
    for corpus_idx in range(corpus.count):
        token_list = corpus.get_token_list(corpus_idx)
        for word in token_list:
            x.append(vectorize_function(word))
            readable_x.append(word)
    return x, readable_x

class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


def index_builder(lst, start_index=1, reverse=False):

    index = dict()

    # Create index dict (reserve zero index for non element in index)
    for idx, key in enumerate(lst, start_index):
        # Duplicate index (multiple key same index)
        if isinstance(key, list):
            for k in key:
                if reverse:
                    index[idx] = k
                else:
                    index[k] = idx

        # Unique index
        else:
            if reverse:
                index[idx] = key
            else:
                index[key] = idx

    return index