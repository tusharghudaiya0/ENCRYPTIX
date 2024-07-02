from sklearn.base import TransformerMixin, BaseEstimator
import re 
import string 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import LancasterStemmer 

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stemmer = LancasterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def cleaning_data(self, text):
        text = text.lower()
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'.pic\S+', '', text)
        text = re.sub(r'[^a-zA-Z+]', ' ', text)
        text = "".join([i for i in text if i not in string.punctuation])
        words = nltk.word_tokenize(text)
        text = " ".join([i for i in words if i not in self.stop_words and len(i) > 2])
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.cleaning_data(text) for text in X]