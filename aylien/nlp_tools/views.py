from nlp_tools import models
from nlp_tools import serializers

import sys
sys.setrecursionlimit(40000)
sys.path.append('../../Thai_NLP_platform')

from bailarn.tokenizer.tokenizer import Tokenizer
from bailarn.word_embedder.word2vec import Word2Vec
from bailarn.ner.ner import NamedEntityRecognizer
from bailarn.pos.pos_tagger import POSTagger
from bailarn.categorization.categorization import Categorization
from bailarn.sentiment.analyzer import SentimentAnalyzer
from bailarn.keyword_expansion.keyword_expansion import KeywordExpansion

from bailarn.utils import utils

from bailarn.tokenizer import constant as tokenizer_constant
from bailarn.word_embedder import constant as word_embedder_constant
from bailarn.ner import constant as ner_constant
from bailarn.pos import constant as pos_constant
from bailarn.categorization import constant as categorization_constant
from bailarn.sentiment import constant as sentiment_constant
from bailarn.keyword_expansion import constant as keyword_expansion_constant

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from keras.preprocessing.sequence import pad_sequences

import numpy as np

from bs4 import BeautifulSoup
import urllib.request
import json
import pickle

# from django.views.decorators.csrf import csrf_exempt

# Tokenizer
tokenizer_model = Tokenizer()
tokenizer_char_index = utils.build_tag_index(
    tokenizer_constant.CHARACTER_LIST, tokenizer_constant.CHAR_START_INDEX)
tokenizer_tag_index = utils.build_tag_index(
    tokenizer_constant.TAG_LIST, tokenizer_constant.TAG_START_INDEX)

# Word2Vec
w2v_model = Word2Vec()

# Categorization
categorization_model = Categorization()
categorization_word_index = json.load(
    open('../../Thai_NLP_platform/bailarn/categorization/categorization_word_index.json'))
categorization_tag_index = utils.build_tag_index(
    categorization_constant.TAG_LIST, categorization_constant.TAG_START_INDEX)

# NER
ner_model = NamedEntityRecognizer()
ner_word_index = pickle.load(
    open('../../Thai_NLP_platform/bailarn/ner/ner_word_index.pickle', 'rb'))
ner_word_index["<PAD>"] = 0
ner_tag_index = utils.build_tag_index(
    ner_constant.TAG_LIST, start_index=ner_constant.TAG_START_INDEX)

# Sentiment
sentiment_model = SentimentAnalyzer()
sentiment_word_index = pickle.load(
    open('../../Thai_NLP_platform/bailarn/sentiment/sentiment_word_index.pickle', 'rb'))
sentiment_word_index.pop('<UNK>', None)
sentiment_word_index['UNK'] = len(sentiment_word_index)
sentiment_tag_index = utils.build_tag_index(
    sentiment_constant.TAG_LIST, sentiment_constant.TAG_START_INDEX)

# POS
pos_model = POSTagger()
pos_word_index = json.load(
    open('../../Thai_NLP_platform/bailarn/pos/pos_word_index.json'))
pos_tag_index = utils.build_tag_index(
    pos_constant.TAG_LIST, start_index=pos_constant.TAG_START_INDEX)
pos_tag_index["<PAD>"] = 0


def tokenize(text):
    return tokenizer_model.predict(text)


# Keyword_expansion
keyword_expansion_model = KeywordExpansion(tokenizer=tokenize)


def tag(model, vector_list):
    result = model.predict([vector_list])
    return result


def pad(raw_y_train, max_num_words):
    padded_y_train = []
    for y in raw_y_train:
        padded_y_train.append((y + ([0] * max_num_words))[0:max_num_words])
    return padded_y_train


def round_up(vector):
    return [format(x, '.2f') for x in vector]


def calculate_distance(v, w):
    return format(np.linalg.norm(v - w), '.2f')


def vectorize(word_list):
    # vectorize each tokens
    vector_list = []
    for w in word_list:
        vector = w2v_model.predict(w)
        vector_list.append(vector)
    return vector_list

    # validate input string length
    if len(input_string) > 1000:
        return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)


from bs4 import BeautifulSoup
import urllib.request


def match_class(target):
    # target = target.split()
    def do_match(tag):
        try:
            classes = dict(tag.attrs)["class"]
        except KeyError:
            classes = ""
        classes = classes.split()
        return all(c in classes for c in target)
    return do_match


def crawl_webpage(url_in):
    with urllib.request.urlopen(url_in) as url:
        html = url.read()
        soup = BeautifulSoup(html, 'html.parser')

   # remove all script and style elements
    for script in soup(['script', 'style']):
        script.extract()  # rip it out

    p_tag_lists = ''
    if 'pantip' in url_in:
        p_tag_lists = soup.findAll("div", class_='display-post-story')[0].text

    else:
        for p_tag in soup.findAll(['p', 'h1', 'h2']):
            t = p_tag.text.replace('\n', '').replace(' ', '')
            if(len(t) >= 200):
                p_tag_lists = p_tag_lists + ' \n' + t

    return p_tag_lists


@api_view(['POST'])
def get_token(request):
    if request.method == 'POST':
        # get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)
            if input_string == "":
                token = models.StringList(
                    string_list=["Sorry, The text from this URL cannot be retrieved."])
                token.save()
                serializer = serializers.StringListSerializer(token)
                return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)

        # reject too long string
        if len(input_string) > 100000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        # tokenize the input
        word_list = tokenize(input_string)
        # serialize output
        token = models.StringList(string_list=word_list)
        token.save()
        serializer = serializers.StringListSerializer(token)

        return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def get_vector(request):
    if request.method == 'POST':
        # get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)
            if input_string == "":
                token = models.StringList(
                    string_list=["Sorry, The text from this URL cannot be retrieved."])
                token.save()
                serializer = serializers.StringListSerializer(token)
                return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)

        # reject too long string
        if len(input_string) > 100000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        # tokenize & vectorize the input
        word_list = tokenize(input_string)
        vector_list = vectorize(word_list)

        # serialize output
        rounded_vector_list = [round_up(x) for x in vector_list]
        word_vector = models.VectorList(
            string_list=word_list, vector_list=rounded_vector_list)
        word_vector.save()
        serializer = serializers.VectorListSerializer(word_vector)
        return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def get_categorization(request):
    if request.method == 'POST':
        # get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)
            if input_string == "":
                token = models.StringList(
                    string_list=["Sorry, The text from this URL cannot be retrieved."])
                token.save()
                serializer = serializers.StringListSerializer(token)
                return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)

        # reject too long string
        if len(input_string) > 100000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        # tokenize&vectorize the input
        texts = utils.TextCollection(tokenize_function=tokenize)
        texts.add_text(input_string)
        vs = utils.build_input(texts, categorization_word_index,
                               categorization_tag_index, categorization_constant.SEQUENCE_LENGTH,
                               target='categorization', for_train=False)
        y = categorization_model.predict(vs.x,
                                         decode_tag=False)
        categorization_inv_map = {v: k for k,
                                  v in categorization_tag_index.items()}
        thershold_selection_dict = json.load(open(
            "../../Thai_NLP_platform/bailarn/categorization/threshold_selection.json"))
        decoded_y_list = []
        confidence_list = []
        for idx, value in enumerate(y[0]):
            thershold = thershold_selection_dict['class_{}'.format(idx)]
            if (value > thershold):
                decoded_y_list.append(categorization_inv_map[idx])
                confidence = (value - thershold)/(1-thershold)
                confidence_list.append(confidence)
        # Norm confidence list to be range of 0 to 1
        confidence_list, decoded_y_list = (list(t) for t in zip(
            *sorted(zip(confidence_list, decoded_y_list), reverse=True)))
        confidence_tag_list = []
        for idx, confidence in enumerate(confidence_list[:5]):
            confidence_tag = models.ConfidenceTag(
                tag=decoded_y_list[idx], confidence=confidence)
            confidence_tag.save()
            confidence_tag_list.append(confidence_tag)
        out = models.ConfidenceTagList(
            confidence_tag_list=confidence_tag_list)
        out.save()
        serializer = serializers.ConfidenceTagListSerializer(out)
        return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def get_sentiment(request):
    if request.method == 'POST':
        # get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)
            if input_string == "":
                token = models.StringList(
                    string_list=["Sorry, The text from this URL cannot be retrieved."])
                token.save()
                serializer = serializers.StringListSerializer(token)
                return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)

        # reject too long string
        if len(input_string) > 100000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        # tokenize&vectorize the input
        texts = utils.TextCollection(tokenize_function=tokenize)
        texts.add_text(input_string)
        vs = utils.build_input(texts, sentiment_word_index,
                               sentiment_tag_index, sentiment_constant.SEQUENCE_LENGTH,
                               target='sentiment', for_train=False)

        sentiment_inv_map = {v: k for k, v in sentiment_tag_index.items()}
        y = sentiment_model.predict(vs.x, decode_tag=False)
        decoded_y_list = []
        confidence_list = []
        for idx, confidence in enumerate(y[0]):
            decoded_y_list.append(sentiment_inv_map[idx])
            confidence_list.append(confidence)

        confidence_list, decoded_y_list = (list(t) for t in zip(
            *sorted(zip(confidence_list, decoded_y_list), reverse=True)))

        confidence_tag_list = []
        for idx, confidence in enumerate(confidence_list):
            confidence_tag = models.ConfidenceTag(
                tag=decoded_y_list[idx], confidence=confidence)
            confidence_tag.save()
            confidence_tag_list.append(confidence_tag)
        out = models.ConfidenceTagList(
            confidence_tag_list=confidence_tag_list)
        out.save()
        serializer = serializers.ConfidenceTagListSerializer(out)
        return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def get_ner(request):
    if request.method == 'POST':

        # get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)
            if input_string == "":
                token = models.StringList(
                    string_list=["Sorry, The text from this URL cannot be retrieved."])
                token.save()
                serializer = serializers.StringListSerializer(token)
                return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)

        # reject too long string
        if len(input_string) > 100000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        # tokenize & vectorize the input
        ner_texts = utils.TextCollection(tokenize_function=tokenize)
        ner_texts.add_text(input_string)
        word_list = ner_texts.get_token_list(0)

        vs = utils.build_input(ner_texts, ner_word_index,
                               ner_tag_index, ner_constant.SEQUENCE_LENGTH,
                               target='ner', for_train=False)

        # remove padding
        y_pred = ner_model.predict(vs.x, decode_tag=True)[0][0:len(word_list)]
        tagged = models.TaggedToken(
            token_list=word_list, tag_list=y_pred)
        tagged.save()
        serializer = serializers.TaggedTokenSerializer(tagged)
        return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def get_pos(request):
    if request.method == 'POST':
        # get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)
            if input_string == "":
                token = models.StringList(
                    string_list=["Sorry, The text from this URL cannot be retrieved."])
                token.save()
                serializer = serializers.StringListSerializer(token)
                return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)

        # reject too long string
        if len(input_string) > 100000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        # tokenize & vectorize the input
        texts = utils.TextCollection(tokenize_function=tokenize)
        texts.add_text(input_string)
        word_list = texts.get_token_list(0)

        vs = utils.build_input(texts, pos_word_index,
                               pos_tag_index, pos_constant.SEQUENCE_LENGTH,
                               target='pos', for_train=False)
        # remove padding
        y_pred = pos_model.predict(vs.x, decode_tag=True).flatten()[
            0:len(word_list)]
        tagged = models.TaggedToken(token_list=word_list, tag_list=y_pred)
        tagged.save()
        serializer = serializers.TaggedTokenSerializer(tagged)
        return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def get_keyword_expansion(request):
    if request.method == 'POST':
        # get input string
        if request.data['type'] == 'raw_text':
            input_string = request.data['text']

        elif request.data['type'] == 'webpage':
            url = request.data['url']
            input_string = crawl_webpage(url)
            if input_string == "":
                token = models.StringList(
                    string_list=["Sorry, The text from this URL cannot be retrieved."])
                token.save()
                serializer = serializers.StringListSerializer(token)
                return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)

        # reject too long string
        if len(input_string) > 100000:
            return Response(status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        # remove padding
        word_similarity_list = keyword_expansion_model.predict(input_string)
        word_list = []
        similarity_list = []
        for word, similarity in word_similarity_list:
            word_list.append(word)
            similarity_list.append(similarity)
        out = models.SimilarityList(
            string_list=word_list, similarity_list=similarity_list)
        out.save()
        serializer = serializers.SimilarityListSerializer(out)
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
        distance_dict = {}
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                # d = calculate_distance(vector_list[i], vector_list[j])
                d = w2v_model.model.wv.similarity(word_list[i], word_list[j])
                distance_dict[(word_list[i], word_list[j])] = d

        for pair, d in sorted(distance_dict.items(), key=lambda x: x[1], reverse=True):
            vd = models.VectorDistance(w1=pair[0], w2=pair[1], distance=d)
            vd.save()
            l.append(vd)

        word_vector = models.VectorDistanceList(
            string_list=word_list, vector_list=rounded_vector_list, distances=l)
        word_vector.save()
        serializer = serializers.VectorDistanceListSerializer(word_vector)
    return Response(serializer.data, status=status.HTTP_200_OK)
