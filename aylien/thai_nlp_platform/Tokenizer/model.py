import deepcut

class Tokenizer(object):
    """ A Tokenizer (implemented by deepcut) :
            input format : any string (e.g. "ฉันกินข้าว")
            output format : list of tokenized string (e.g. ["ฉัน", "กิน", "ข้าว"])
    """
    def __init__(self):
        self.model = deepcut

    def predict(self, sentence):
        return self.model.tokenize(sentence)