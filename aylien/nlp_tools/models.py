from django.db import models


class StringList(models.Model):
    string_list = models.TextField()

class VectorList(models.Model):
    string_list = models.TextField()
    vector_list = models.TextField()


class SimilarityList(models.Model):
    string_list = models.TextField()
    similarity_list = models.TextField()


class ConfidenceTag(models.Model):
    tag = models.TextField()
    confidence = models.FloatField()


class ConfidenceTagList(models.Model):
    confidence_tag_list = models.TextField()


class TaggedToken(models.Model):
    token_list = models.TextField()
    tag_list = models.TextField()


class VectorDistance(models.Model):
    w1 = models.TextField()
    w2 = models.TextField()
    distance = models.TextField()


class VectorDistanceList(models.Model):
    string_list = models.TextField()
    vector_list = models.TextField()
    distances = models.TextField()
