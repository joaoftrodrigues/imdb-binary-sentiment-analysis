from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd 
import preTrained_models

# Read test data
test_ds = pd.read_csv('../data/imdb_reviews_test.csv')

# Label using TextBlob
predicted_labels = preTrained_models.predict_dataset_labels(test_ds['text'], 'vaderSentiment')

# Accuracy obtained from TextBlob
tb_acc = preTrained_models.accuracy_score(predicted_labels, test_ds['label'])

# Show accuracy
print(tb_acc)