import nltk 
import pandas as pd
import lexicon_tools

# Read test data
test_ds = pd.read_csv('/data/imdb_reviews_test.csv')

# Tokenize
test_ds['tokenize'] = [nltk.word_tokenize(text) for text in test_ds['text']] 

# Label reviews based on lexicon analysis
predicted_labels = lexicon_tools.lexicon_analysis(test_ds['tokenize'])

# Accuracy obtained
accuracy = lexicon_tools.models_tools.accuracy_score(predicted_labels, test_ds['label'])

print(accuracy)