from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer


def fit_vectorize_bow(texts):
    """ Create vectorize model of Bag of Words to apply on texts """

    # Count vectorizer for bag of words
    count_vect = CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3), lowercase=False)

    # Fit model with train texts
    count_vect_model = count_vect.fit(texts)

    return count_vect_model


def fit_vectorize_tfidf(texts):
    """ Create vectorize model of TF-IDF to apply on text """

    # TF-IDF vectorizer
    tf_vectorizer = TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3), lowercase=True)

    # Fit model with train texts
    tf_vectorizer_model = tf_vectorizer.fit(texts)

    return tf_vectorizer_model


def binarize_labels(train_texts):
    """ Create model to binarize labels => Convert it to 0 and 1 """

    # Label object
    lb = LabelBinarizer()

    # Label model fit with train data
    lb_model = lb.fit(train_texts)

    return lb_model
