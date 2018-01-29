from nlp_tools import models
from nlp_tools import serializers

from thai_nlp_platform.Tokenizer.model import Tokenizer 
from thai_nlp_platform.Word2Vec.model import WordEmbedder
from thai_nlp_platform.ner.model import NamedEntityRecognizer
from thai_nlp_platform.POS.pos_tagger import POSTagger
from thai_nlp_platform.utils import utils
from thai_nlp_platform.utils import co as constant
from thai_nlp_platform.utils import loader
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from keras.preprocessing.sequence import pad_sequences
# from django.views.decorators.csrf import csrf_exempt
tokenizer = Tokenizer()
word_embedder = WordEmbedder()
ner_model = NamedEntityRecognizer(model_path='thai_nlp_platform/ner/models/test.h5',max_num_words = 500, word_vec_length=100)
pos_model = POSTagger(model_path='thai_nlp_platform/POS/models/model.h5')

def tokenize(text):
    return tokenizer.predict(text)

def vectorize(word):    
    return word_embedder.predict(word)

def tag(model, vector_list):
    result = model.predict([vector_list])
    return result
def pad(raw_y_train, max_num_words):
    padded_y_train = []
    for y in raw_y_train :
        padded_y_train.append((y+([0]*max_num_words))[0:max_num_words])
    return padded_y_train
@api_view(['POST'])
def get_token(request):
    if request.method == 'POST':
        word_list = tokenize(request.data['text'])
        token = models.Token(word_list=word_list)
        token.save()
        serializer = serializers.TokenSerializer(token);
        return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['POST'])
def get_vector(request):
    if request.method == 'POST':
        word_list = tokenize(request.data['text'])
        vector_list = []
        for w in word_list:
            vector_list.append(vectorize(w))
        word_vector = models.WordVector(vector_list=vector_list)
        word_vector.save()
        serializer = serializers.WordVectorSerializer(word_vector)
        return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['POST'])
def get_ner(request):
    if request.method == 'POST':
        word_list = tokenize(request.data['text'])
        text_len = len(word_list)
        vector_list = []
        for w in word_list:
            vector_list.append(vectorize(w).tolist())
        x = pad_sequences([vector_list],maxlen=500,value=[0]*100)
        y_pred = [ner_model.predict(x)[0][0:text_len]]
        rev_tag_index = utils.index_builder(constant.NE_LIST, start_index=1, reverse=True)
        y_pred_decode = loader.decode_tag(y_pred, rev_tag_index)
        tagged = models.TaggedToken(token_list = word_list,ner_list=y_pred_decode[0])
        tagged.save()
        serializer = serializers.TaggedTokenSerializer(tagged)
        return Response(serializer.data, status=status.HTTP_200_OK)
import numpy as np
@api_view(['POST'])
def get_pos(request):
    if request.method == 'POST':
        word_list = tokenize(request.data['text'])
        text_len = len(word_list)
        vector_list = []
        for w in word_list:
            vector_list.append(vectorize(w).tolist())
        x = pad_sequences([vector_list],maxlen=500,value=[0]*100)
        y_pred = pos_model.predict(x)
        rev_tag_index = utils.index_builder(constant.TAG_LIST, start_index=1, reverse=True)
        y = []
        for tag_idx in y_pred['pos_tag'][0:text_len]:
            y.append(rev_tag_index[tag_idx])
        tagged = models.TaggedToken(token_list = word_list,tag_list=y)
        tagged.save()
        serializer = serializers.TaggedTokenSerializer(tagged)
        return Response(serializer.data, status=status.HTTP_200_OK)
