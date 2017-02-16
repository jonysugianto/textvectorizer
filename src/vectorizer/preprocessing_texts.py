from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from collections import defaultdict
import logging

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

manual_stopwords=['sex', 'porn', 'https', 'http']


def remove_stopwords(tokens):
    tokens = [t for t in tokens if not t in en_stop]
    return tokens


def remove_manual_stopwords(tokens):
    tokens = [t for t in tokens if not t in manual_stopwords]
    return tokens


def make_filter_rare_words(frequency_tables, min_number_of_word):
    def filter(t):
        if frequency_tables[t] >= min_number_of_word:
            return True
        else:
            return False

    return filter


def remove_rare_words(tokens, word_frequency, min_number_of_word):
    return filter(make_filter_rare_words(word_frequency, min_number_of_word), tokens)


def remove_one_symbol_token(tokens):
    return filter(lambda t: len(t)>1, tokens)


def isnumber(str):
    try:
        float(str)
        return True
    except Exception as e:
        #print(e)
        return False

def remove_number(tokens):
    return filter(lambda t: not isnumber(t), tokens)


def decode_and_tokenize(txt):
#    decoded_txt=txt.lower().decode("utf-8")
    decoded_txt=txt.lower()
    tokens=tokenizer.tokenize(decoded_txt)
    return tokens


def filter_tokens(tokens):
    filtered_tokens=remove_stopwords(tokens)
    filtered_tokens=remove_number(filtered_tokens)
    filtered_tokens=remove_one_symbol_token(filtered_tokens)
    filtered_tokens=remove_manual_stopwords(filtered_tokens)
    return filtered_tokens


def preprocess_text(txt):
    tokens=decode_and_tokenize(txt)
    filtered_tokens=filter_tokens(tokens)
    return filtered_tokens


def preprocess_listoftext(listoftxt):
    listoftokens=[]
    for txt in listoftxt:
        tokens=preprocess_text(txt)
        listoftokens.append(tokens)
    return listoftokens
