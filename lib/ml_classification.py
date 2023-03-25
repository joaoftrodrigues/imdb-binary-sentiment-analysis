from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression


def multinomial_nb_train(train_features, train_labels):
    """ Train model of Multinomial Naive Bayes """

    # Create object to use Multinomial Naive Bayes
    multi_nb = MultinomialNB()

    #fitting the svm for bag of words
    multi_nb = multi_nb.fit(train_features, train_labels)

    return multi_nb


def svm_train(train_features, train_labels, kernel='linear'):
    """ Train model of SVM """

    svm_model = SVC(kernel=kernel)

    svm_model = svm_model.fit(train_features, train_labels)

    return svm_model


def logistic_reg(train_features, train_labels):
    """ Train model of Logistic Regression """

    logistic_reg = LogisticRegression()

    logistic_reg = logistic_reg.fit(train_features, train_labels)

    return logistic_reg


