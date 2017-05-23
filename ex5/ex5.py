import nltk
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import *
from scipy.sparse import csr_matrix
import unicodedata
import sys
import pickle
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

'''
conversao de caracteres maiusculos para minusculos
remoçao de pontuaçao
remoçao de stop words
steming dos termos

'''

NUM_DOCUMENTS = 5000
SVD_NUM_COMPONENTS = 2440

# Lower case all text and remove punctuation
def get_tokens(fname):
    with open(fname, 'r') as text:
        lowers = text.read().lower()

        # remove punctuation
        punctuation_tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                                        if unicodedata.category(chr(i)).startswith('P'))
        no_punctuation = lowers.translate(punctuation_tbl)
        no_punctuation = no_punctuation.translate(str.maketrans('','',string.punctuation))

        tokens = nltk.word_tokenize(no_punctuation)
        return tokens


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def get_terms(fname):
    # tokens
    tokens = get_tokens(fname)

    # remove stop words
    filtered = [w for w in tokens if not w in stopwords.words('english')]

    # stem
    stemmed = stem_tokens(filtered)

    return stemmed


def get_data(binary=False):
    indptr = [0]
    indices = []
    data_binary = []
    data_tf = []
    vocabulary = {}

    # for each document
    for doc_index in range(1, NUM_DOCUMENTS + 1):
        fname = 'ex5_files/raw/' + str(doc_index) + '.occurences.npy'

        with open(fname, 'rb') as handle:
            term_occurences = pickle.load(handle)
            for term, occurences in term_occurences.items():
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                data_binary.append(1)
                data_tf.append(occurences)

            indptr.append(len(indices))

    X = csr_matrix((data_binary, indices, indptr), dtype=int).toarray()
    mask = np.all([X.sum(0) != NUM_DOCUMENTS, X.sum(0) > 1], axis=0)
    print(X.shape)
    if not binary:
        X = csr_matrix((data_tf, indices, indptr), dtype=int).toarray()
    X = X[:, mask]
    print(X.shape)

    y = pd.read_csv('ex5-category.tab', sep=' ', names=['doc', 'category'], header=0)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y['category'].head(NUM_DOCUMENTS))

    return X, y

def run_binary():
    X, y = get_data(binary=True)
    clf = BernoulliNB()
    scores = cross_val_score(clf, X, y, cv=5) # by default this will use stratified k-fold
    print("------------------- BernoulliNB --------------------")
    print("Accuracies", scores)
    print("Bernoulli NB accuracy:", scores.mean())

def run_multi():
    X, y = get_data(binary=False)
    clf = MultinomialNB()
    scores = cross_val_score(clf, X, y, cv=5) # by default this will use stratified k-fold
    print("------------------- MultinomialNB --------------------")
    print("Accuracies", scores)
    print("Multinomial NB accuracy:", scores.mean())

def run_all():
    X, y = get_data(binary=False)
    svm_accs = []
    rf_accs = []
    gb_accs = []
    outter = StratifiedKFold(n_splits=5, random_state=1)

    for train_index, test_index in outter.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reduce dimensionality to 90% of variance
        svd = TruncatedSVD(n_components=SVD_NUM_COMPONENTS)
        X_train = svd.fit_transform(X_train)
        X_test = svd.transform(X_test)
        print("SVD variance:", svd.explained_variance_ratio_.sum())

        svm_accs.append(run_svm(X_train, X_test, y_train, y_test))
        rf_accs.append(run_rf(X_train, X_test, y_train, y_test))
        gb_accs.append(run_gb(X_train, X_test, y_train, y_test))

        print('----------------------------')

    print("------------------- TruncatedSVD + SVM --------------------")
    print(svm_accs)
    print('mean accuracy:', "%.3f" % np.mean(svm_accs))

    print("------------------- TruncatedSVD + RF --------------------")
    print(rf_accs)
    print('mean accuracy:', "%.3f" % np.mean(rf_accs))

    print("------------------- TruncatedSVD + GB --------------------")
    print(gb_accs)
    print('mean accuracy:', "%.3f" % np.mean(gb_accs))


def run_svm(X_train, X_test, y_train, y_test):
    print("Starting SVM")

    # find best hyperparams
    gammas = 2 ** np.array([-15.0, -10.0, -5.0, 0.0, 5.0])
    costs = 2 ** np.array([-5.0, 0.0, 5.0, 10.0])
    parameters = {'C': costs, 'gamma': gammas}
    clf = GridSearchCV(svm.SVC(kernel='rbf', decision_function_shape='ovr'), parameters, cv=3)
    clf.fit(X_train, y_train)

    # train with the best hyperparams
    svr = svm.SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
    svr.fit(X_train, y_train)
    return svr.score(X_test, y_test)


def run_rf(X_train, X_test, y_train, y_test):
    print("Starting RF")

    # find best hyperparams
    parameters = {'max_features': (2, 3, 5, 7), 'n_estimators': (100, 200, 400, 800)}
    clf = GridSearchCV(RandomForestClassifier(random_state=1), parameters, cv=3)
    clf.fit(X_train, y_train)

    # train with the best hyperparams
    clf = RandomForestClassifier(random_state=1,
                                 n_estimators=clf.best_params_['n_estimators'],
                                 max_features=clf.best_params_['max_features'])
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def run_gb(X_train, X_test, y_train, y_test):
    print("Starting GB")

    # find best hyperparams
    parameters = {'n_estimators': (30, 70, 100), 'learning_rate': (0.1, 0.05)}
    clf = GridSearchCV(GradientBoostingClassifier(max_depth=5, random_state=1), parameters, cv=3)
    clf.fit(X_train, y_train)

    # train with the best hyperparams
    clf = GradientBoostingClassifier(max_depth=5, random_state=1,
                                     n_estimators=clf.best_params_['n_estimators'],
                                     learning_rate=clf.best_params_['learning_rate'])
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


if sys.argv[1] == 'preprocess':
    terms = get_terms()
    count = Counter(terms)
    import pickle
    with open(sys.argv[2].replace('.txt','') + '.occurences.npy', 'wb') as handle:
        pickle.dump(count, handle, protocol=pickle.HIGHEST_PROTOCOL)
elif sys.argv[1] == 'binary':
    run_binary()
elif sys.argv[1] == 'multi':
    run_multi()
elif sys.argv[1] == 'all':
    run_all()