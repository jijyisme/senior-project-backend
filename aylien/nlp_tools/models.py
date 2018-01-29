from django.db import models
import json

class Token(models.Model):
    word_list = models.TextField()

class WordVector(models.Model):
	vector_list = models.TextField()

class TaggedToken(models.Model):
	token_list = models.TextField()
	tag_list = models.TextField()
