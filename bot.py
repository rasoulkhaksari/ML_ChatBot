import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Bot(metaclass=Singleton):

    def __init__(self, model, words, classes, intents) -> None:
        self.model = model
        self.words = words
        self.classes = classes
        self.intents = intents
        self.lemmatizer = WordNetLemmatizer()

    def bow(self, sentence):
        sentence_words = [self.lemmatizer.lemmatize(word) for word in word_tokenize(sentence.lower()) if word.isalpha()]
        bag = [0]*len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
        return(np.array(bag))

    def predict_class(self, sentence):  # -> list:
        X = self.bow(sentence)
        prediction_class = self.model.predict(X.reshape(1, -1))[0]
        prediction_tag = self.classes[prediction_class]
        return prediction_tag

    def response(self, msg):
        prediction_tag = self.predict_class(msg)
        for intent in self.intents:
            if(intent['tag'] == prediction_tag):
                return np.random.choice(intent['responses'])
        return ''
