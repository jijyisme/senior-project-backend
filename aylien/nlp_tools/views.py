from nlp_tools import models
from nlp_tools import serializers

from thai_nlp_platform.Tokenizer.model import Tokenizer 
from thai_nlp_platform.Word2Vec.model import WordEmbedder
from thai_nlp_platform.ner.model import NamedEntityRecognizer

from thai_nlp_platform.utils import utils
from thai_nlp_platform.utils import co as constant
from thai_nlp_platform.utils import loader
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
# from django.views.decorators.csrf import csrf_exempt


def tokenize(text):
    tokenizer = Tokenizer()
    word_list = tokenizer.predict(text)
    return word_list

def vectorize(word_list):
    print('word_list', word_list)
    word_embedder = WordEmbedder()
    vector_list = word_embedder.predict(word_list)
    print('vector_list',vector_list)
    return vector_list

def tag(model, vector_list):
    result = model.predict([vector_list])
    return result

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
        vector_list = vectorize(word_list)
        word_vector = models.WordVector(vector_list=vector_list)
        word_vector.save()
        serializer = serializers.WordVectorSerializer(word_vector)
        return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['POST'])
def get_ner(request):
    if request.method == 'POST':
        corpus = utils.TextCollection(corpus_directory='./data/BEST_mock/',
                              tokenize_function = None, 
                              word_delimiter='|', 
                              tag_delimiter='/', 
                              tag_dictionary = {'word': 0,'pos': 1,'ner': 2} 
                             )
        tag_index = utils.index_builder(constant.NE_LIST, constant.TAG_START_INDEX)
        rev_tag_index = utils.index_builder(constant.NE_LIST, constant.TAG_START_INDEX, reverse=True)
        vs = utils.build_input(corpus=corpus,tag_index=tag_index,num_step=300
                               ,vectorize_function=WordEmbedder().predict,needed_y='ner')
        print('shape',vs.x.shape)
        print(vs.x)
        # ner = NamedEntityRecognizer(model_path='thai_nlp_platform/ner/models/ner005.h5',max_num_words = 300, word_vec_length=100)
        # y_pred = ner.predict(vs.x)
        # y_pred_decode = loader.decode_tag(y_pred, rev_tag_index)
        # tagged = models.Token(tag_list=y_pred_decode)
        # tagged.save()
        # serializer = serializers.TokenSerializer(tagged)
        # return Response(serializer.data, status=status.HTTP_200_OK)


