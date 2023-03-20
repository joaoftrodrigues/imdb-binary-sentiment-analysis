import nltk 
import pandas as pd
from lib import models_tools

    
def remove_entities(texts):
    """ Remove entities from tokenized text """

    updated_texts = []

    for text in texts:

        # Apply POS-Tagging
        tagged_text = nltk.pos_tag(text)

        tokens_without_entities = [tagged_token[0] for tagged_token in tagged_text 
         if tagged_token[1] != 'NNP']

        updated_texts.append(tokens_without_entities)

    return updated_texts


def lexicon_analysis(texts, filepath='lexicons/NCR-lexicon.csv'):
    """ Apply labels based on lexicon """

    # Import NCR-lexicon
    ncr_lexicon = pd.read_csv(filepath)

    # Set Words as index and remove id column
    ncr_lexicon.set_index('English', inplace=True)

    # Resume sentiment values to single column, with values -1, 0, 1
    lex = ncr_lexicon['Positive'] - ncr_lexicon['Negative']

    # Transform to Dictionary, for easier manipulation
    lex = lex.to_dict()

    #lemmatizer = nltk.stem.WordNetLemmatizer()
    predicted_labels = []

    for text in texts:

        text_polarity = 0
        for word in text:

            # Reduce word to its lemma
            #lemma = lemmatizer.lemmatize(word)

            # Get polarity from dictionary
            word_polarity = lex.get(word) 

            # Add when is not None
            if word_polarity:
              text_polarity += word_polarity        

        # Add text label to list, based on value
        predicted_labels.append(models_tools.polarity_to_label(text_polarity))

    return predicted_labels
