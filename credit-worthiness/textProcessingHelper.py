import string
import nltk
import re
from nltk.stem import WordNetLemmatizer


# Remove punctuation
def remove_punctuation(text):
    clean_text = "".join([char for char in text if char not in string.punctuation])
    return clean_text


# Tokenizing
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


# Remove stop words
stop_word = nltk.corpus.stopwords.words('english')


def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stop_word]
    return text


# Lemmatizing
wn = nltk.WordNetLemmatizer()


def lemmatizing(tokenized_list):
    text = [wn.lemmatize(word) for word in tokenized_list]
    return text


# Noise Removal
def scrub_words(text):
    text = re.sub("_", "", text)
    text = re.sub("(\\W|\\d)", " ", text)
    text = text.strip()
    return text


def label_words(key, tokenized_list):
    return tokenized_list.count(key)


