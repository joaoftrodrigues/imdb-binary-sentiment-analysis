import nltk 
import pandas as pd
from lib import models_tools
import string

    
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


def lexicon_analysis(texts, filepath='lexicons/NCR-lexicon.csv'):
    """ Apply labels based on lexicon """

    # Initialize NCR lexicon and transform to use
    lex = ncr_initializer(filepath)

    # Initialize lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Store output labels
    predicted_labels = []

    # Remove punctuation
    texts_without_punctuation = remove_punctuation(texts)

    # Remove entities
    texts_without_entities = remove_entities(texts_without_punctuation)

    for text in texts_without_entities:

        # To sum polarity
        text_polarity = 0

        for word in text:

            # Reduce word to its lemma
            lemma = lemmatizer.lemmatize(word)

            # Get polarity from dictionary
            word_polarity = lex.get(lemma) 

            # Add when is not None
            if word_polarity:
              text_polarity += word_polarity        

        # Add text label to list, based on value
        predicted_labels.append(models_tools.polarity_to_label(text_polarity))

    return predicted_labels

def remove_punctuation(texts):
    """ Remove punctuation tokens """

    texts_no_punctuation = []

    for text in texts: 

        # Catch tokens that are not punctuation
        texts_no_punctuation.append(
            [token for token in text if token not in string.punctuation]
        )
            
    #[[token for token in text if token not in string.punctuation] for text in texts ]

    return texts_no_punctuation
