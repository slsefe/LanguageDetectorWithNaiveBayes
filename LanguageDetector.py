import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB


class LanguageDetector():

    def __init__(self, classifier=MultinomialNB(), vectorize = 'N-gram'):
        self.classifier = classifier
        if vectorize == 'N-gram':
            self.vectorizer = CountVectorizer(ngram_range=(1,2), max_features=1000, preprocessor=self._remove_noise)
        elif vectorize == 'TF-IDF':
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000, preprocessor=self._remove_noise)
    
    def _remove_noise(self, document):
        noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+", "\#\w*\#"]))
        clean_text = re.sub(noise_pattern, "", document)
        return clean_text

    def features(self, X):
        '''Transform documents to document-term matrix.'''
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        '''Fit Naive Bayes classifier according to X, y.'''
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        '''Perform classification on an array of test vectors X.'''
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        '''Returns the mean accuracy on the given test data and labels.'''
        return self.classifier.score(self.features(X), y)
