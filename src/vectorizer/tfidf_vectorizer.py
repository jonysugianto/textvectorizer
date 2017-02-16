from gensim import models
from vectorizer.preprocessing_texts import preprocess_listoftext, preprocess_text

def build_TfIdfModel_from_list_of_texts(list_of_texts, dict):
    list_of_text_tokens=preprocess_listoftext(list_of_texts)
    list_of_texts_bow = [dict.doc2bow(ts) for ts in list_of_text_tokens]
    list_of_texts_bow=filter(lambda x: len(x)>1, list_of_texts_bow)
    tfidf_model = models.TfidfModel(list_of_texts_bow)
    return tfidf_model


def convert_text_to_tfidf(text, dict, tfidf_model):
    tokens=preprocess_text(text)
    bow=dict.doc2bow(tokens)
    tfid=tfidf_model[bow]
    return tfid


def convert_list_of_texts_to_list_of_tfidf(list_of_texts, dict, tfidf_model):
    list_of_tfidf=[]
    for txt in list_of_texts:
        list_of_tfidf.append(convert_text_to_tfidf(txt, dict, tfidf_model))
    return list_of_tfidf


def generate_keywords_from_text_with_tfidf(text, dict, tfidf_model, max_num_keywords):
    tfidf_vector=convert_text_to_tfidf(text, dict, tfidf_model)
    sorted_score_tfidf = sorted(tfidf_vector, key=lambda item: item[1], reverse=True)
    keyword_indexs=map(lambda item:item[0], sorted_score_tfidf)
    #keywords=map(lambda item:dict.get(item), keyword_indexs)
    keywords=[]
    for ki in keyword_indexs:
        keywords.append(dict.get(ki))

    if len(keywords)<max_num_keywords:
        return keywords
    else:
        return keywords[:max_num_keywords]


def save_tfidf_model(tfidf_model, filename):
    try:
        tfidf_model.save(filename)
    except Exception as e:
        print('error on saving TfIdf model:', e)


def load_tfidf_model(filename):
    try:
        return models.TfidfModel.load(filename)
    except Exception as e:
        print('error on Loading TFIDF model:', e)
        return None
