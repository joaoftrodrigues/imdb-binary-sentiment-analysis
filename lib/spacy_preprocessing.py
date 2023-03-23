import spacy 
import contextualSpellCheck


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

        # Present entities in current text        
        entities_found = [str(entity) for entity in text_features.ents]

        for token in text_features:

            # Numerals and Punctuation
            if token.pos in ['NUM', 'PUNCT']:
                continue

            # Entities
            if token.text in entities_found:
                continue

            # Add lowercase token to list
            processed_text.append(token.text.lower())
        
        # Add tokens to work, from text
        processed_texts.append(processed_text)
