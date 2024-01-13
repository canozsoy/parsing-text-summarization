import os
import re
import nltk
from nltk.corpus import stopwords
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

def read_from_json_file(filename):
    file = open(filename, "r")
    megadoc = json.load(file)
    file.close()
    return megadoc

def write_to_json_file(filename, content):
    json_file = json.dumps(content)
    file = open(filename, "w")
    file.write(json_file)

def read_from_pickle_file(filename):
    file = open(filename, "rb")
    vectorizer = pickle.load(file)
    file.close()
    return vectorizer

def save_to_pickle_file(filename, content):
    file = open(filename, "wb")
    pickle.dump(content, file)
    file.close()

def get_all_filenames_in_dir(dir_path):
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

def read_and_preprocess(filename, dirname):
    f = open(dirname + "/" + filename, "r")
    raw_file = f.read()
    f.close()
    matched = re.findall("<sentence\s+id=\"[\w\d]+\">([^<>]+)<\/sentence>", raw_file)
    document = ""
    sentences = []
	
    for match in matched:
        tokens = nltk.tokenize.word_tokenize(match)
        sentence = []
        for token in tokens:
            lowered_token = token.lower()
            if lowered_token.isalpha() and lowered_token not in stop_words:
                sentence.append(lowered_token)
        joined_sentence = " ".join(sentence)
        sentences.append(joined_sentence)
        document += joined_sentence
    
    return {"document": document, "sentences": sentences}

def get_megadoc():
    megadoc_filename = "megadoc.json"

    if os.path.exists(megadoc_filename):
        return read_from_json_file(megadoc_filename)
    
    megadoc = {}
    
    files_dirname = "Task2"
    filenames = get_all_filenames_in_dir(files_dirname)

    for filename in filenames:
        # Below files are corrupted or not utf-8 encoded
        if filename in ["06_1261.xml", "06_782.xml", "09_585.xml", "06_1718.xml"]:
            continue
        doc = read_and_preprocess(filename, files_dirname)
        filename_without_extension = filename.replace(".xml", "")
        megadoc[filename_without_extension] = {"document": doc["document"], "sentences": doc["sentences"]}

    write_to_json_file(megadoc_filename, megadoc)

    return megadoc

def merge_documents(megadoc):
    documents = []
    for key in megadoc:
        item = megadoc[key]["document"]
        documents.append(item)
    return documents

def get_tfidf_vectorizer(megadoc):
    vectorizer_filename = "vectorizer.pickle"

    if os.path.exists(vectorizer_filename):
        return read_from_pickle_file(vectorizer_filename)
    
    documents = merge_documents(megadoc)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)

    save_to_pickle_file(vectorizer_filename, vectorizer)
    return vectorizer

def get_ifidf_dict_for_word(text, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    tfidf_matrix= vectorizer.transform([text]).todense()
    feature_index = tfidf_matrix[0,:].nonzero()[1]
    tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
    return dict(tfidf_scores).get(text, 0)

def get_cosine_similarity(vectorizer, document):
    document_query = vectorizer.transform(document)
    cosine_similarities = cosine_similarity(document_query, document_query)
    return cosine_similarities.mean(axis=1)

def access_with_fallback_value(array, index):
    return int(array[index]) if len(array) >= index + 1 else ""

def prepare_top_five_dict(cosine_similarities):
    index_sorted = (-cosine_similarities).argsort()[:5]
    return {"first": access_with_fallback_value(index_sorted, 0),
             "second": access_with_fallback_value(index_sorted, 1),
               "third": access_with_fallback_value(index_sorted, 2),
                 "forth": access_with_fallback_value(index_sorted, 3),
                   "fifth": access_with_fallback_value(index_sorted, 4)}
    
def get_top_five_sentence_dict(megadoc, vectorizer):
    top_five_dict_filename = "top-five-dict.json"

    if os.path.exists(top_five_dict_filename):
        return read_from_json_file(top_five_dict_filename)
    
    top_five_dict = {}
    
    for key in megadoc:
        item = megadoc[key]["sentences"]
        cosine_similarities = get_cosine_similarity(vectorizer, item)
        current_top_five_dict = prepare_top_five_dict(cosine_similarities)
        top_five_dict[key] = current_top_five_dict
    
    write_to_json_file(top_five_dict_filename, top_five_dict)

    return top_five_dict

def print_summary(top_five_dict, document_name, megadoc):
    indexes = top_five_dict[document_name]
    all_sentences = []

    for index_text in indexes:
        index = indexes[index_text]
        all_sentences.append(megadoc[document_name]["sentences"][index])
    
    result = "\n".join(all_sentences)
    print("Summary for document" + document_name + "\n" + result)

def main():
    document_name = "06_28"
    megadoc = get_megadoc()
    vectorizer = get_tfidf_vectorizer(megadoc)
    top_five_dict = get_top_five_sentence_dict(megadoc, vectorizer)
    print_summary(top_five_dict, document_name, megadoc)

if __name__ == "__main__":
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))
    main()