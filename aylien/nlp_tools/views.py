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

import numpy as np

from bs4 import BeautifulSoup
import urllib.request

# from django.views.decorators.csrf import csrf_exempt
tokenizer = Tokenizer()
word_embedder = WordEmbedder()
ner_model = NamedEntityRecognizer(model_path='thai_nlp_platform/ner/models/test.h5',max_num_words = 500, word_vec_length=100)
pos_model = POSTagger(model_path='thai_nlp_platform/POS/models/model.h5')

def tokenize(text):
    return tokenizer.predict(text)

def tag(model, vector_list):
    result = model.predict([vector_list])
    return result
def pad(raw_y_train, max_num_words):
    padded_y_train = []
    for y in raw_y_train :
        padded_y_train.append((y+([0]*max_num_words))[0:max_num_words])
    return padded_y_train

def round_up(vector):
    return [format(x, '.2f') for x in vector]

def calculate_distance(v, w):
    return format(np.linalg.norm(v-w),'.2f')

def vectorize(word_list):
    #vectorize each tokens
    vector_list = []
    for w in word_list:
        vector = word_embedder.predict(w)
        vector_list.append(vector)
    return vector_list


    # #validate input string length
    # if len(input_string) > 1000:
    #     return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

def crawl_webpage(url_in):
    with urllib.request.urlopen(url_in) as url:
        html = url.read()
        soup = BeautifulSoup(html, 'html.parser')
   
   # remove all script and style elements
    for script in soup(['script', 'style']):
        script.extract()  # rip it out
    
    p_tag_lists=''
    for p_tag in soup.findAll('p','div'):
        t = p_tag.text
        if(len(t) >= 250):
            p_tag_lists=p_tag_lists+'\n'+t
    print('crawled words',p_tag_lists)
    return p_tag_lists

@api_view(['POST'])
def get_token(request):
    if request.method == 'POST':
        #get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)

        #reject too long string
        if len(input_string) > 100000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        #tokenize the input
        word_list = tokenize(input_string)
        #serialize output
        token = models.StringList(string_list=word_list)
        token.save()
        serializer = serializers.StringListSerializer(token);
        return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['POST'])
def get_vector(request):
    if request.method == 'POST':
        #get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)
            
        #reject too long string
        if len(input_string) > 1000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        #tokenize&vectorize the input
        word_list = tokenize(input_string)
        vector_list = vectorize(word_list)

        #serialize output
        rounded_vector_list = [round_up(x) for x in vector_list]
        word_vector = models.VectorList(string_list = word_list,vector_list=rounded_vector_list)
        word_vector.save()
        serializer = serializers.VectorListSerializer(word_vector)
        return Response(serializer.data, status=status.HTTP_200_OK)



@api_view(['POST'])
def get_ner(request):
    if request.method == 'POST':
        #get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)
            
        #reject too long string
        if len(input_string) > 1000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
            
        #tokenize&vectorize the input
        word_list = tokenize(input_string)
        text_len = len(word_list)
        vector_list = vectorize(word_list)
        x = pad_sequences([vector_list],maxlen=500,value=[0]*100)
        y_pred = [ner_model.predict(x)[0][0:text_len]]
        rev_tag_index = utils.index_builder(constant.NE_LIST, start_index=1, reverse=True)
        y_pred_decode = loader.decode_tag(y_pred, rev_tag_index)
        tagged = models.TaggedToken(token_list = word_list,tag_list=y_pred_decode[0])
        tagged.save()
        serializer = serializers.TaggedTokenSerializer(tagged)
        return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['POST'])
def get_pos(request):
    if request.method == 'POST':
        #get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)
            
        #reject too long string
        if len(input_string) > 1000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
            
        #tokenize&vectorize the input
        word_list = tokenize(input_string)
        text_len = len(word_list)
        vector_list = vectorize(word_list)

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

@api_view(['GET'])
def get_tag_list(request):
    if request.method == 'GET':
        model_name = request.GET.get('model', '')
        if(model_name == 'POS'):
            token = models.StringList(string_list=constant.TAG_LIST)
            token.save()
        elif(model_name == 'NER'):
            token = models.StringList(string_list=constant.NE_LIST)
            token.save()          
        serializer = serializers.StringListSerializer(token)
        return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['POST'])
def get_distance_list(request):
    if request.method == 'POST':
        word_list = request.data['word_list'][1:-1].split(',')
        word_list = [x.strip() for x in word_list]
        vector_list = vectorize(word_list)
        rounded_vector_list = [round_up(x) for x in vector_list]
        l = []
        for i in range(len(word_list)):
            for j in range(i+1,len(word_list)):
                d = calculate_distance(vector_list[i],vector_list[j])
                vd = models.VectorDistance(w1 = word_list[i], w2 = word_list[j], distance = d)
                vd.save()
                l.append(vd)

        word_vector = models.VectorDistanceList(string_list=word_list,vector_list=rounded_vector_list,distances=l)
        word_vector.save()
        serializer = serializers.VectorDistanceListSerializer(word_vector)
    return Response(serializer.data, status=status.HTTP_200_OK)
