import pandas as pd
import os
import re
import nltk
import string
import emoji
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter


def cleansing(text):
    # remove non-ascii
    text = text.encode('ascii', 'ignore').decode('ascii')

    emoticons = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [D\)\]\(\]/\\OpP] # Mouth
        )"""
    # remove emoticon
    text = re.sub(emoticons, '', text)

    # remove URLs
    text = re.sub(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        '', text)

    # remove punctuations
    text = re.sub(r'[^\w]|_', ' ', text)

    # remove digit from string
    text = re.sub("\S*\d\S*", "", text).strip()

    # remove digit or numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # to lowercase
    text = text.lower()

    # Remove additional white spaces
    text = re.sub('[\s]+', ' ', text)

    return text


def createStopword():
    punctuation = list(string.punctuation)
    stopwordsSastrawi = StopWordRemoverFactory().get_stop_words()
    stopwordsNtlk = nltk.corpus.stopwords.words('indonesian')
    stopwords = stopwordsNtlk + stopwordsSastrawi + punctuation + ['rt', 'via', '…', '•']

    return stopwords

def removeStopword(text):
    text = ' '.join(word for word in text.split() if word not in stopwords)

    return text


def stemmerFactory(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)

    return text


def preProcessing(text):
    text = cleansing(text)
    text = removeStopword(text)
    text = stemmerFactory(text)

    return text

if __name__ == '__main__':
    data = pd.read_csv('./data/contoh3.csv',delimiter='|')
    stopwords = createStopword()
    newText = [i.split() for i in data['text'].apply(preProcessing)]