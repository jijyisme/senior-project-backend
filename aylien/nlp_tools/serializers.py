from rest_framework import serializers
from nlp_tools.models import Token, WordVector


class TokenSerializer(serializers.Serializer):
    # id = serializers.IntegerField(read_only=True)
    word_list = serializers.ListField(child=serializers.CharField(), min_length=None, max_length=None)

    def create(self, validated_data):
        """
        Create and return a new `Snippet` instance, given the validated data.
        """
        return Token.objects.create(**validated_data)

    def update(self, instance, validated_data):
        """
        Update and return an existing `Snippet` instance, given the validated data.
        """
        instance.word_list = validated_data
        instance.save()
        return instance

class WordVectorSerializer(serializers.Serializer):
    # id = serializers.IntegerField(read_only=True)
    vector_list = serializers.ListField(child=(serializers.ListField(child=serializers.FloatField(), min_length=None, max_length=None)),min_length=None, max_length=None)

    def create(self, validated_data):
        """
        Create and return a new `Snippet` instance, given the validated data.
        """
        return WordVector.objects.create(**validated_data)


    # def get_vector_list(self, instance):
    #     return instance.get_vector_list()

    def update(self, instance, validated_data):
        """
        Update and return an existing `Snippet` instance, given the validated data.
        """
        instance.vector_list = validated_data
        instance.save()
        return instance
