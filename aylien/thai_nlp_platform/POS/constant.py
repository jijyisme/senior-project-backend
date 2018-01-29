"""
Global Constant
"""

import string

# Spacebar
SPACEBAR = " "

# Escape Character
ESCAPE_WORD_DELIMITER = "\t"
ESCAPE_TAG_DELIMITER = "\v"

# data dimention
DATA_DIM = 100

# Tag
PAD_TAG_INDEX = 0
NON_SEGMENT_TAG_INDEX = 1
TAG_START_INDEX = 2

# BEST
TAG_LIST = ["NN", "NR", "PPER", "PINT", "PDEM", "DPER", "DINT", "DDEM", "PDT","REFX", "VV", "VA", "AUX", "JJA", "JJV", "ADV", "NEG", "PAR", "CL","CD", "OD", "FXN", "FXG", "FXAV", "FXAJ", "COMP", "CNJ", "P", "IJ","PU", "FWN", "FWV", "FWA", "FWX"]   

# ORCHID
# TAG_LIST = ["NPRP" ,"NCNM","NONM","NLBL","NCMN","NTTL","PPRS","PDMN","PNTR","PREL","VACT","VSTA","VATT","XVBM","XVAM","XVMM","XVBB","XVAE","DDAN","DDAC","DDBQ","DDAQ","DIAC","DIBQ","DIAQ","DCNM" ,"DONM" ,"ADVN" ,"ADVI" ,"ADVP" ,"ADVS","CNIT" ,"CLTV","CMTR","CFQC" ,"CVBL" ,"JCRG" ,"JCMP" ,"JSBR" ,"RPRE" ,"INT" ,"FIXN" ,"FIXV" ,"EAFF" ,"EITT" ,"NEG" ,"PUNC" ]

NUM_TAGS = len(TAG_LIST) + 2

DEFULT_MODEL_PATH = "./checkpoint/24-10-2017-14-03-32/0011-0.1125.hdf5"

# Random Seed
SEED = 1395096092
