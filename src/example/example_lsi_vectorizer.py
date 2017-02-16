from example.example_data import documents
from vectorizer.dictionary_builder import load_dictionary
from vectorizer.tfidf_vectorizer import load_tfidf_model
from vectorizer.lsi_vectorizer import save_lsi_model, convert_text_to_lsi, build_LsiTopicModel

file_dictionary = '/tmp/example.dict'
file_tfidf_model = '/tmp/example.tfidf'
file_lsi_model = '/tmp/example.lsi'
dict=load_dictionary(file_dictionary)
tfidf_model=load_tfidf_model(file_tfidf_model)

lsi_model=build_LsiTopicModel(documents,file_dictionary,file_tfidf_model, 3)
save_lsi_model(file_lsi_model, lsi_model)

sample="Graph for Human machine interface"
lsi=convert_text_to_lsi(sample, dict, tfidf_model, lsi_model)
print(sample, lsi)