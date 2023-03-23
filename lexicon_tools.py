import nltk 
import pandas as pd
from lib import models_tools, preprocessing, spacy_preprocessing

    
def ncr_initializer(filepath):
    """ Reads NCR file and transforms it in dict
        of word / score """

    # Import NCR-lexicon
    ncr_lexicon = pd.read_csv(filepath)

    # Set Words as index and remove id column
    ncr_lexicon.set_index('English', inplace=True)

    # Resume sentiment values to single column, with values -1, 0, 1
    lex = ncr_lexicon['Positive'] - ncr_lexicon['Negative']

    # Transform to Dictionary, for easier manipulation
    lex = lex.to_dict()

    return lex

def nltk_preprocessing(texts):

    # Remove punctuation
    texts_without_punctuation = preprocessing.remove_punctuation(texts)

    # Remove entities
    texts_without_entities = preprocessing.remove_entities(texts_without_punctuation)

    # Delete numbers
    deleted_numbers = preprocessing.delete_numbers(texts_without_entities)

    # Lower tokens
    lowered_texts = preprocessing.lower_texts(texts_without_entities)

    return lowered_texts


def lexicon_analysis(texts, filepath='lexicons/NCR-lexicon.csv'):
    """ Apply labels based on lexicon """

    # Initialize NCR lexicon and transform to use
    lex = ncr_initializer(filepath)

    # Store output labels
    predicted_labels = []

    # Preprocessed texts, using Spacy
    preprocessed_texts = spacy_preprocessing.preprocessing(texts)

    for text in preprocessed_texts:

        # To sum polarity
        text_polarity = 0

        for word in text:
            
            # Get polarity from dictionary
            word_polarity = lex.get(word) 

            # Add when is not None
            if word_polarity:
              text_polarity += word_polarity        

        # Add text label to list, based on value
        predicted_labels.append(models_tools.polarity_to_label(text_polarity))

    return predicted_labels
