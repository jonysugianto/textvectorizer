
from example.example_data import documents
from vectorizer.dictionary_builder import build_dictionary,save

output_file_dictionary = '/tmp/example.dict'
dict = build_dictionary(documents)
save(dict, output_file_dictionary)
print(dict)

