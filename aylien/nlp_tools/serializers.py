from rest_framework import serializers
from nlp_tools.models import Token, WordVector


class TokenSerializer(serializers.Serializer):
    word_list = serializers.ListField(child=serializers.CharField(), min_length=None, max_length=None)

    def create(self, validated_data):
        return Token.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.word_list = validated_data
        instance.save()
        return instance

class WordVectorSerializer(serializers.Serializer):
    vector_list = serializers.ListField(child=(serializers.ListField(child=serializers.FloatField(), min_length=None, max_length=None)),min_length=None, max_length=None)

    def create(self, validated_data):
        return WordVector.objects.create(**validated_data)
    def update(self, instance, validated_data):
        instance.vector_list = validated_data
        instance.save()
        return instance

class TaggedTokenSerializer(serializers.Serializer):
    token_list = serializers.ListField(child=serializers.CharField(), min_length=None, max_length=None)
    tag_list = serializers.ListField(child=serializers.CharField(), min_length=None, max_length=None)
    def create(self, validated_data):
        return WordVector.objects.create(**validated_data)
    def update(self, instance, validated_data):
        instance.token_list = validated_data
        instance.tag_list = validated_data
        instance.save()
        return instance