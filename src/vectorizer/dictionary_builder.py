from gensim import corpora
from vectorizer.preprocessing_texts import preprocess_listoftext


def build_dictionary(list_of_texts):
    list_of_tokens=preprocess_listoftext(list_of_texts)
    dict=corpora.Dictionary(list_of_tokens)
    return dict


def save(dict, dict_filename):
    try:
        dict.save(dict_filename)
    except Exception as e:
        print('error on saving dictionary:', e)


def load_dictionary(dict_filename):
    try:
        dict=corpora.Dictionary.load(dict_filename)
        return dict
    except Exception as e:
        print('error on loading dictionary:', e)
        return None
