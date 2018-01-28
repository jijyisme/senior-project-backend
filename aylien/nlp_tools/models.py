from django.db import models
import json

class Token(models.Model):
    word_list = models.TextField()

class WordVector(models.Model):
	vector_list = models.TextField()
    # def get_vector_list(self):
    # 	return self.vector_list