import nltk 
import pandas as pd 
import string 


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

    punctuation = string.punctuation + '...'
    for text in texts: 

        # Catch tokens that are not punctuation
        texts_no_punctuation.append(
            [token for token in text if token not in punctuation]
        )
            
    #[[token for token in text if token not in string.punctuation] for text in texts ]

    return texts_no_punctuation

def lower_texts(texts):
    """ Lower tokens of all texts """

    return [[token.lower() for token in text] for text in texts]


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