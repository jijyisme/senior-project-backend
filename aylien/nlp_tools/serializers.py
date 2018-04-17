from rest_framework import serializers
from nlp_tools.models import StringList, VectorList, VectorDistanceList, ConfidenceTagList


class StringListSerializer(serializers.Serializer):
    string_list = serializers.ListField(
        child=serializers.CharField(), min_length=None, max_length=None)

    def create(self, validated_data):
        return Token.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.string_list = validated_data
        instance.save()
        return instance


class SimilarityListSerializer(serializers.Serializer):
    string_list = serializers.ListField(
        child=serializers.CharField(), min_length=None, max_length=None)
    similarity_list = serializers.ListField(
        child=serializers.FloatField(), min_length=None, max_length=None)

    def create(self, validated_data):
        return SimilarityList.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.string_list = validated_data
        instance.similarity_list = validated_data
        instance.save()
        return instance


class ConfidenceTagSerializer(serializers.Serializer):
    tag = serializers.CharField()
    confidence = serializers.FloatField()


class ConfidenceTagListSerializer(serializers.ModelSerializer):
    confidence_tag_list = serializers.ListField(
        child=ConfidenceTagSerializer(),  min_length=None, max_length=None)

    def create(self, validated_data):
        return ConfidenceTagList.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.confidence_tag_list = validated_data
        instance.save()
        return instance

    class Meta:
        model = ConfidenceTagList
        fields = ['confidence_tag_list']


class VectorListSerializer(serializers.Serializer):
    string_list = serializers.ListField(
        child=serializers.CharField(), min_length=None, max_length=None)
    vector_list = serializers.ListField(child=(serializers.ListField(child=serializers.FloatField(
    ), min_length=None, max_length=None)), min_length=None, max_length=None)

    def create(self, validated_data):
        return WordVector.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.string_list = validated_data
        instance.vector_list = validated_data
        instance.save()
        return instance


class TaggedTokenSerializer(serializers.Serializer):
    token_list = serializers.ListField(
        child=serializers.CharField(), min_length=None, max_length=None)
    tag_list = serializers.ListField(
        child=serializers.CharField(), min_length=None, max_length=None)

    def create(self, validated_data):
        return TaggedToken.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.token_list = validated_data
        instance.tag_list = validated_data
        instance.save()
        return instance


class VectorDistanceSerializer(serializers.Serializer):
    w1 = serializers.CharField()
    w2 = serializers.CharField()
    distance = serializers.FloatField()


class VectorDistanceListSerializer(serializers.ModelSerializer):
    string_list = serializers.ListField(
        child=serializers.CharField(), min_length=None, max_length=None)
    vector_list = serializers.ListField(child=(serializers.ListField(child=serializers.FloatField(
    ), min_length=None, max_length=None)), min_length=None, max_length=None)
    distances = serializers.ListField(
        child=VectorDistanceSerializer(),  min_length=None, max_length=None)

    def create(self, validated_data):
        return VectorDistanceList.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.string_list = validated_data
        instance.vector_list = validated_data
        instance.distances = validated_data
        instance.save()
        return instance

    class Meta:
        model = VectorDistanceList
        fields = ['string_list', 'vector_list', 'distances']
