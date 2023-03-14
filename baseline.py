import pandas as pd 
import bibl
import os 

## Read data ##

if os.path.exists('cache_results/textBlob_predicted.csv'):
    test_ds = pd.read_csv('cache_results/textBlob_predicted.csv')
else:  
    # Path of testing data
    testData_file = 'data/imdb_reviews_test.csv'

    # Import data and assign manually header
    test_ds = pd.read_csv(testData_file)

    # Get polarity, using TextBlob
    test_ds['predicted_label_tb'] = bibl.get_dataset_polarities(test_ds['text'])  
    test_ds.to_csv('cache_results/textBlob_predicted.csv')

test_ds['tb_labels'] = bibl.label_polarities(test_ds['predicted_label_tb'])

accuracy = bibl.accuracy_score(test_ds['tb_labels'], test_ds['label'])

print(f"Accuracy: {accuracy}")
