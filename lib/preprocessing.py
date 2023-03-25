import nltk 
import pandas as pd 
import string 
import re
from bs4 import BeautifulSoup
import spacy
from nltk.tokenize.toktok import ToktokTokenizer


def remove_entities(texts):
    """ Remove entities from tokenized text """

    updated_texts = []

    for text in texts:

        # Apply POS-Tagging
        tagged_text = nltk.pos_tag(text)

        # Keep tokens that are not entities
        tokens_without_entities = [tagged_token[0] for tagged_token in tagged_text 
         if tagged_token[1] != 'NNP']

        # Add previous tokens' list
        updated_texts.append(tokens_without_entities)

    return updated_texts


def remove_punctuation(texts):
    """ Remove punctuation tokens """

    texts_no_punctuation = []

    for text in texts: 

        # Catch tokens that are not punctuation
        texts_no_punctuation.append(
            [re.sub(f"[{string.punctuation}][{string.punctuation}]+",'',token) for token in text if token != '...']
        )
            
    #[[token for token in text if token not in string.punctuation] for text in texts ]

    return texts_no_punctuation


def remove_extra_punctuation(texts):
    """ Remove punctuation tokens """

    punct_no_dot = string.punctuation.replace('.','')

    texts_no_punctuation = [re.sub(f"[{punct_no_dot}][{punct_no_dot}]+",'',text) for text in texts]

    texts_no_punctuation = [re.sub("[\.]{4,}",'',text) for text in texts_no_punctuation]

    return texts_no_punctuation


def lower_texts_tokens(texts):
    """ Lower tokens of all texts """

    return [[token.lower() for token in text] for text in texts]


def delete_numbers(texts):
    """ Delete tokens that represent numbers """

    for text in texts:
        for token in text:
            if any(char.isdigit() for char in token):
                text.remove(token)

    return texts


# TODO: Finish Long Words function
def long_words(texts):
    """Long Words treatment"""

    for text in texts:
        for token in text:
            prev_char = ""
            count = 1
            start_index = 0
            for i in range(len(token)):
                if token[i] == prev_char:
                    count += 1
                    if count == 4:
                        print(token)
                else:
                    prev_char = token[i]
                    count = 1
                    start_index = i


def remove_tags_from_texts(texts):

    untagged_texts = []

    for text in texts:
        soup = BeautifulSoup(text, 'html.parser')
        untagged_texts.append(soup.get_text())

    return untagged_texts


def fast_preprocessing(texts):

    texts_no_tags = remove_tags_from_texts(texts)

    treated_text = remove_extra_punctuation(texts_no_tags)

    return treated_text


def remove_special_characters(text):
    """ Let only letters """

    pattern=r'[^a-zA-z\s]'
    text=re.sub(pattern,'',text)
    return text


def remove_stopwords(texts):

    en_model = spacy.load('en_core_web_sm')
    stopwords = en_model.Defaults.stop_words

    tokenizer = ToktokTokenizer()

    processed_texts = []
    
    for text in texts:

        # Tokenize text
        tokenized_text = tokenizer.tokenize(text)

        # Remove stopwords
        tokens_no_stopwords = [token for token in tokenized_text if token.lower() not in stopwords]

        text_no_stopwords = ' '.join(tokens_no_stopwords)
        # Append current text to output list
        processed_texts.append(text_no_stopwords)

    return processed_texts


def lemmatization(texts):

    # Load model
    nlp = spacy.load('en_core_web_sm')
    
    # Use only lemmatizer
    #nlp.pipeline = ['lemmatizer']

    total = len(texts)
    n_docs = 1
    # Use pipe for efficiency
    texts = nlp.pipe(texts)

    texts_lemma = []

    for text in texts:

        # Get tokens' lemma on current text
        tokens_lemma = [token.lemma_ for token in text]

        # Join tokens in single string
        tokens_str = ' '.join(tokens_lemma)

        # Add string to output list
        texts_lemma.append(tokens_str)
        print(f"{n_docs}/{total}")
        n_docs += 1

    return texts_lemma 
        
    
def ml_preprocessing(texts):
    
    # Remove HTML tags
    #processed_texts = remove_tags_from_texts(texts)

    # Remove Special characters
    #processed_texts = [remove_special_characters(text) for text in texts]

    # Remove stopwords
    processed_texts = remove_stopwords(texts)

    #processed_texts = lemmatization(processed_texts)

    return processed_texts


    


