from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

def evaluate(y_true, y_pred):
	y_true = np.array(y_true).flatten()
	y_pred = np.array(y_pred).flatten()
	eva_score = {}
	eva_score['acc'] = accuracy_score(y_true, y_pred)
	eva_score['f1_macro'] = f1_score(y_true, y_pred, average='macro')  
	eva_score['f1_micro'] = f1_score(y_true, y_pred, average='micro')  
	eva_score['precision_macro'] = precision_score(y_true, y_pred, average='macro')
	eva_score['precision_micro'] = precision_score(y_true, y_pred, average='micro')
	eva_score['recall_macro'] = recall_score(y_true, y_pred, average='macro')
	eva_score['recall_micro'] = recall_score(y_true, y_pred, average='micro')
	return eva_score

