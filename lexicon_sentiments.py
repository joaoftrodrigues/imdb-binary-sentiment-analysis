import nltk 
import pandas as pd
import lexicon_tools
from lib import spacy_preprocessing

# Read test data
test_ds = pd.read_csv('data/imdb_reviews_test.csv')

# Label reviews based on lexicon analysis
predicted_labels = spacy_preprocessing.preprocess_and_evaluation_nocorr(test_ds['text'])

# Accuracy obtained
accuracy = lexicon_tools.models_tools.accuracy_score(predicted_labels, test_ds['label'])

print(accuracy)