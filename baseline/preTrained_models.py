from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import stanza 
from ..lib import models_tools

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
        return [models_tools.polarity_to_label(tb_sentiment(text)) for text in texts]
    
    # Vader Sentiment
    elif model == 'vaderSentiment':
        vader_analyzer = SentimentIntensityAnalyzer()
        return [models_tools.polarity_to_label(vader_sentiment_withObject(vader_analyzer, text)) for text in texts]
    
    elif model == 'stanza':
        stanza_analyzer = stanza.Pipeline(lang='en', processors='tokenize,sentiment', tokenize_no_ssplit=False)

        # Values from stanze ranges from 0 to 2;
        # by subtracting 1, it gets normalized with other models (-1 to 1)
        return [models_tools.polarity_to_label(stanza_sentiment(stanza_analyzer, text)-1) for text in texts]

    else:
        return 0
    