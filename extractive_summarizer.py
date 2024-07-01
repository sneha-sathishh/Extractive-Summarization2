# Required Libraries
# import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import pymupdf

# Download stopwords if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

def summarizer(file_path,keywords:str,lines:int): 
# Preprocessing Function
    def preprocess_text(text):
        stop_words = set(stopwords.words('english'))
        sentences = sent_tokenize(text)
        cleaned_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in stop_words]
            cleaned_sentences.append(words)
        return cleaned_sentences, sentences

    # Function to get sentence vector
    def get_sentence_vector(sentence, model):
        words = [word for word in word_tokenize(sentence.lower()) if word in model.wv.key_to_index]
        if not words:
            return np.zeros(model.vector_size)
        return np.mean([model.wv[word] for word in words], axis=0)
    
    # Compute sentence vectors
    # sentence_vectors = np.array([get_sentence_vector(sentence, model) for sentence in original_sentences])


    # Keyword similarity function
    def rank_sentences_by_keyword(keyword, sentence_vectors, original_sentences, model):
        keyword_vector = get_sentence_vector(keyword, model).reshape(1, -1)
        similarities = cosine_similarity(keyword_vector, sentence_vectors).flatten()
        ranked_indices = [index for index, _ in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)]
        ranked_sentences = [original_sentences[i] for i in ranked_indices]
        return ranked_sentences, ranked_indices


    # Summarize based on keyword
    def summarize(text, keyword, lines):
        cleaned_sentences, original_sentences = preprocess_text(text)
        model = Word2Vec(cleaned_sentences, vector_size=100, window=5, min_count=1, workers=4)
        sentence_vectors = np.array([get_sentence_vector(sentence, model) for sentence in original_sentences])
        ranked_sentences, ranked_indices = rank_sentences_by_keyword(keyword, sentence_vectors, original_sentences, model)
        top_indices = sorted(ranked_indices[:lines])
        summary = ' '.join([original_sentences[i] for i in top_indices])
        return summary

    extracted_text = ""
    doc = pymupdf.open(file_path) # open a document
    for page in doc: # iterate the document pages
        extracted_text += page.get_text() # get plain text encoded as UTF-8

    #     # Get cleaned sentences and original sentences
    # cleaned_sentences, original_sentences = preprocess_text(extracted_text)
        
    #     #  Train Word2Vec Model
    # model = Word2Vec(cleaned_sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    summary = summarize(extracted_text, keywords ,lines)
    return summary

