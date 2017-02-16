from example.example_data import documents
from vectorizer.dictionary_builder import load_dictionary
from vectorizer.tfidf_vectorizer import build_TfIdfModel_from_list_of_texts, save_tfidf_model, convert_text_to_tfidf

file_dictionary = '/tmp/example.dict'
output_tfidf_model_filename = '/tmp/example.tfidf'
dict=load_dictionary(file_dictionary)
tfidf_model=build_TfIdfModel_from_list_of_texts(documents, dict)
save_tfidf_model(tfidf_model, output_tfidf_model_filename)

sample="Graph for Human machine interface"
tfidf=convert_text_to_tfidf(sample, dict, tfidf_model)
print(sample, tfidf)