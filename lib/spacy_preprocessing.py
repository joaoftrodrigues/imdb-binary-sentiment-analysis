import spacy 
import contextualSpellCheck
from lib import models_tools
import pandas as pd 


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


def preprocessing(texts):
    """ Preprocessing using spacy """

    # Load model
    nlp = spacy.load('en_core_web_sm')

    # Add grammar correction
    nlp.add_pipe('contextual spellchecker')

    # Store texts after processing
    processed_texts = []

    for text in texts:

        # First processing is for spell check
        # This way, lemma comes from the corrected word
        corrected_text = nlp(text)._.outcome_spellCheck

        # Retrieval of information from spacy
        text_features = nlp(corrected_text)

        # List of tokens to work from current text
        processed_text = []

        for token in text_features:

            # Numerals
            if token.pos == 'NUM':
                continue

            # Entities
            if token.ent_type:
                continue

            # Add lowercase token to list
            processed_text.append(token.lemma_)
        
        # Add tokens to work, from text
        processed_texts.append(processed_text)

    return processed_texts

    
def preprocess_and_evaluation_nocorr(texts, filepath='lexicons/NCR-lexicon.csv'):
    """ Preprocessing + label using spacy """

    # Load model
    nlp = spacy.load('en_core_web_sm')

    # Store texts after processing
    predicted_labels = []

    # Initialize NCR lexicon and transform to use
    lex = ncr_initializer(filepath)

    total = len(texts)
    n_text = 0

    for text in texts:
        
        # Retrieval of information from spacy
        text_features = nlp(text)

        text_polarity = 0

        for token in text_features:

            # Numerals
            if token.pos == 'NUM':
                continue

            # Entities
            if token.ent_type:
                continue       

            # Polarity of word, from dictionary
            word_polarity = lex.get(token.lemma_)

            # Add when is not None
            if word_polarity:
                text_polarity += word_polarity    
        
        # Add tokens to work, from text
        predicted_labels.append(models_tools.polarity_to_label(text_polarity))
        n_text += 1
        print(f"{n_text}/{total}")

    return predicted_labels