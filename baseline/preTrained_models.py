from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import stanza 


def tb_sentiment(text):
    """ Returns raw value from TextBlob polarity """

    return TextBlob(text).sentiment.polarity

def vader_sentiment_withObject(obj, text):
    """ Returns raw value from Vader Sentiment polarity.
     Implies there's a vaderSentiment object """
    return obj.polarity_scores(text)['compound']


def stanza_sentiment(obj, text):
    """ Returns raw value from Stanza polarity """

    # Used index 0, because each input outputs a single value
    return obj(text).sentences[0].sentiment


def get_dataset_polarities(texts, model):
    """ Returns polarity for each review, on a list """

    # TextBlob
    if model == 'textBlob':
        return [tb_sentiment(text) for text in texts ]
    
    # Vader Sentiment
    elif model == 'vaderSentiment':
        vader_analyzer = SentimentIntensityAnalyzer()
        return [vader_sentiment_withObject(vader_analyzer, text) for text in texts]


def predict_dataset_labels(texts, model):
    """ Attribute a binary label to input texts, based 
        on TextBlob polarities """
    
    # TextBlob
    if model == 'textBlob':
        return [polarity_to_label(tb_sentiment(text)) for text in texts]
    
    # Vader Sentiment
    elif model == 'vaderSentiment':
        vader_analyzer = SentimentIntensityAnalyzer()
        return [polarity_to_label(vader_sentiment_withObject(vader_analyzer, text)) for text in texts]
    
    elif model == 'stanza':
        stanza_analyzer = stanza.Pipeline(lang='en', processors='tokenize,sentiment', tokenize_no_ssplit=False)

        # Values from stanze ranges from 0 to 2;
        # by subtracting 1, it gets normalized with other models (-1 to 1)
        return [polarity_to_label(stanza_sentiment(stanza_analyzer, text)-1) for text in texts]

    else:
        return 0

    

def polarity_to_label(polarity):
    """ Converts polarity into label 
        If value 0 -> maintains 0 
    """
    
    if polarity > 0:
        return "pos"
    
    else:
        return "neg"
    

def label_polarities(polarities):
    """ Assign a label to each polarity value. 
        0 keeps 0.
    """

    return [polarity_to_label(polarity) for polarity in polarities]

def accuracy_score(predicted, ground_truth):
    """ Returns accuracy of predictions """

    predicted = list(predicted)
    ground_truth = list(ground_truth)

    # Auxiliar variable to count corrects
    corrects_counter = 0

    total_elements = len(predicted)

    # Check if both lists have same length
    if total_elements != len(ground_truth):
        return -1 
    
    # Count correct predictions
    for i in range(total_elements):
        if predicted[i] == ground_truth[i]:
             corrects_counter += 1
        
    print(corrects_counter)
    # Calculus of accuracy
    accuracy = corrects_counter / total_elements
    return accuracy


