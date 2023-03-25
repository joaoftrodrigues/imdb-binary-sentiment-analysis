import pandas as pd 
from lib import preprocessing, spacy_preprocessing, models_tools, features, ml_classification
from sklearn.metrics import accuracy_score


# Read Data
test_data = pd.read_csv('data/imdb_reviews_test.csv')
train_data = pd.read_csv('data/imdb_reviews_train.csv')

# Preprocess data
train_data['text'] = preprocessing.ml_preprocessing(train_data['text'])
test_data['text'] = preprocessing.ml_preprocessing(test_data['text'])

# Features
## Vectorize texts
vector_model = features.fit_vectorize_tfidf(train_data['text'])

train_data_vector = vector_model.transform(train_data['text'])
test_data_vector = vector_model.transform(test_data['text'])

## Binarize labels
label_bin_model = features.binarize_labels(train_data['label'])

train_data['label'] = label_bin_model.transform(train_data['label'])
test_data['label'] = label_bin_model.transform(test_data['label'])

# Train model
#class_model = ml_classification.multinomial_nb_train(train_data_vector, train_data['label'])
class_model = ml_classification.logistic_reg(train_data_vector, train_data['label'])

# Evaluate model
test_predictions = class_model.predict(test_data_vector)
mnb_acc = accuracy_score(test_data['label'], test_predictions)
print(mnb_acc)