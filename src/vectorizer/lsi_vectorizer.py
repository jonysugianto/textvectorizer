from vectorizer.dictionary_builder import load_dictionary
from vectorizer.tfidf_vectorizer import load_tfidf_model, \
    convert_list_of_texts_to_list_of_tfidf, convert_text_to_tfidf
from gensim import models


def build_LsiTopicModel(list_of_texts, dict_filename, tfidf_filename,  number_topics):
    try:
        dict=load_dictionary(dict_filename)
        tfidf_model=load_tfidf_model(tfidf_filename)
        list_of_tfidf=convert_list_of_texts_to_list_of_tfidf(list_of_texts, dict, tfidf_model)
        lsi_model = models.LsiModel(list_of_tfidf, id2word=dict, num_topics=number_topics)
        return lsi_model
    except Exception as e:
        print('error on creating Lsi Model:', e)
        return None


def convert_text_to_lsi(text, dict, tfidf_model, lsi_model):
    tfidf=convert_text_to_tfidf(text, dict, tfidf_model)
    lsi=lsi_model[tfidf]
    return lsi


def convert_list_of_texts_to_list_of_lsi(list_of_texts, dict_filename, tfidf_filename,
                     lsi_model_filename):
    try:
        dict= load_dictionary(dict_filename)
        tfidf_model = load_tfidf_model(tfidf_filename)
        lsi_model = load_lsi_model(lsi_model_filename)
        list_of_lsi=[]
        for txt in list_of_texts:
            lsi = convert_text_to_lsi(txt, dict, tfidf_model, lsi_model)
            list_of_lsi.append(lsi)
        return list_of_lsi
    except Exception as e:
        print('error on building Lsi Corpus:', e)
        return None


def save_lsi_model(output_filename, lsi_model):
    try:
        lsi_model.save(output_filename)
    except Exception as e:
        print('Error on saving lsi model', e)


def load_lsi_model(input_filename):
    try:
        lsi = models.LsiModel.load(input_filename)
        return lsi
    except Exception as e:
        print('Error on loading lsi model', e)
        return None
