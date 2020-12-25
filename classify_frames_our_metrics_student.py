# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

import numpy as np

TRAIN_FILE = Path("./raw_data/GunViolence/train.tsv")
DEV_FILE = Path("./raw_data/GunViolence/dev.tsv")
TEST_FILE = Path("./raw_data/GunViolence/test.tsv")

LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# These frames/labels correspond to
# 1) Gun or 2nd Amendment rights
# 2) Gun control/regulation
# 3) Politics
# 4) Mental health
# 5) School or public space safety
# 6) Race/ethnicity
# 7) Public opinion
# 8) Society/culture
# 9) Economic consequences

def load_data_file(data_file):
    """Load newsframing data

    Returns
    -------
    tuple
        First element is a list of strings(headlines)
        If `data_file` has labels, the second element
        will be a list of labels for each headline.
        Otherwise, the second element will be None.
    """
    print("Loading from {} ...".format(data_file.name), end="")
    text_col = "news_title"
    theme1_col = "Q3 Theme1"

    with open(data_file, encoding="utf8") as f:
        df = pd.read_csv(f, sep="\t")
        X = df[text_col].tolist()

        y = None
        if theme1_col in df.columns:
            y = df[theme1_col].tolist()

        print(
            "loaded {} lines {} labels ... done".format(
                len(X), "with" if y is not None else "without"
            )
        )
    return (X, y)

def build_naive_bayes():
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    nb_pipeline = None
    
    nb_pipeline = Pipeline([
     ('vect', CountVectorizer()),
     ('clf', ComplementNB()),
     ])
    
    return nb_pipeline

def build_logistic_regr():
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    logistic_pipeline = None

    logistic_pipeline = Pipeline([
     ('vect', CountVectorizer()),
     ('clf', LogisticRegression()), 
     ])
    
    return logistic_pipeline

def build_svm_pipeline():
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    svm_pipeline = None

    svm_pipeline = Pipeline([
     ('vect', CountVectorizer()),
     ('clf', SGDClassifier()),
     ])

    return svm_pipeline

def build_own_pipeline() -> Pipeline:
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    nn_pipeline = None

    nn_pipeline = Pipeline([
      ('vect', CountVectorizer()),
      ('tfidf', TfidfTransformer()),
      ('clf', MLPClassifier()) 
      ])
    
    return nn_pipeline

def output_predictions(pipeline):
  
    #load TEST_FILE into X_test and y_test
    X_test, y_test = load_data_file(TEST_FILE) 
    assert(y_test is None) 

    prediction = pipeline.predict(X_test) 

    # Makes dataframe and writes to TSV
    filename = 'predictions.tsv' #name of the file (including extension)
    df = pd.DataFrame(prediction) #creates dataframe

    #writes to csv
    df.to_csv(filename, sep="\t", index=False, header=False)

def make_onehot(y, labels):

    labels = set(labels)
    if len(y.shape) != 1:
        raise Exception("Currently support only 1d input to make_onehot")

    # loop through the labels and create a dictionary entry for each label and it's corresponding col number
    label_indices = {label: i for i, label in enumerate(labels)}

    #loop through the y, and label each y with the row it corresponds to using append
    row_selector = [i for i, curr_label in enumerate(y) if curr_label in labels]

    #use label_indices to get the column that the y belongs to and label each y with the col it corresponds to
    column_selector = [label_indices[label] for label in y if label in label_indices]
    
    #creates an empty numpy matrix of size len(y) x len(labels)
    onehot = np.zeros((len(y), len(labels)), dtype=int) 
    
    #fills in the numpy matrix with 1's at the spots of row x col
    onehot[row_selector, column_selector] = 1 

    return onehot

def check_metric_args(y_true, y_pred, average, labels):

    if average not in ["macro", "micro", None]:
        raise Exception("average param must be one of 'macro' or 'micro', or None.")

    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise Exception("shape of y_true and y_pred is not the same")

    return y_true, y_pred


#Each row of the matrix represents the instances in a predicted class while each column represents 
# the instances in an actual class (or vice versa) 
def get_confusion_matrix(y_true, y_pred, labels):
    m = len(y_true)
    n = len(labels)

    d = dict(zip(labels, range(n)))

    confusion_matrix = np.zeros([n,n])

    for i in range(m):
        gold = d[y_true[i]]
        pred = d[y_pred[i]]
        confusion_matrix[pred, gold] += 1

    return confusion_matrix

def precision(y_true, y_pred, average, labels):
    """Calculate precision.

    `labels` will be used to

    Arguments
    ---------
        y_true: list-like
        y_pred: list-like, of same shape as y_true
        average: One of "micro", "macro", or None
        labels: The labels for which we will calculate metrics

    Returns
    -------
        np.ndarray or float:
            If `average` is None, it returns a numpy array of shape
            (len(labels), ), where the precision values for each class are
            calculated.
            Otherwise, it returns either the macro, or micro precision value as
            float.
    """
    y_true, y_pred = check_metric_args(y_true, y_pred, average, labels)

    # At this point, you can be sure that y_true and y_pred are one hot encoded.
    result = None
    m = len(y_true)
    n = len(labels)

    #call get_confusion_matrix function and put the result in confusion_matrix
    confusion_matrix = get_confusion_matrix(y_true, y_pred, labels)

    #compute the result if using micro-averages
    if average == "micro":
        numerator = np.trace(confusion_matrix)
        denominator = np.sum(confusion_matrix)
        result = numerator/denominator

    #compute the precision independently for each class and then take the average 
    elif average == "macro":
        diag = np.diag(confusion_matrix)
        row_sums = np.sum(confusion_matrix, axis=1)
        row_sums_adjusted = np.array([1 if val == 0 else val for val in row_sums])
        result = np.mean(diag/row_sums_adjusted)

    else:
        diag = np.diag(confusion_matrix)
        row_sums = np.sum(confusion_matrix, axis=1)
        row_sums_adjusted = np.array([1 if val == 0 else val for val in row_sums])
        result = diag/row_sums_adjusted

    return result

def recall(y_true, y_pred, average, labels):
    """Calculate recall.

    `labels` will be used to

    Arguments
    ---------
        y_true: list-like
        y_pred: list-like, of same shape as y_true
        average: One of "micro", "macro", or None
        labels: The labels for which we will calculate metrics

    Returns
    -------
        np.ndarray or float:
            If `average` is None, it returns a numpy array of shape
            (len(labels), ), where the recall values for each class are
            calculated.
            Otherwise, it returns either the macro, or micro recall value as
            float.
    """

    y_true, y_pred = check_metric_args(y_true, y_pred, average, labels)

    result = None

    m = len(y_true)
    n = len(labels)

    confusion_matrix = get_confusion_matrix(y_true, y_pred, labels).T

    if average == "micro":
        numerator = np.trace(confusion_matrix)
        denominator = np.sum(confusion_matrix)
        result = numerator/denominator

    elif average == "macro":
        diag = np.diag(confusion_matrix)
        row_sums = np.sum(confusion_matrix, axis=1)
        row_sums_adjusted = np.array([1 if val == 0 else val for val in row_sums])
        result = np.mean(diag/row_sums_adjusted)

    else:

        diag = np.diag(confusion_matrix)
        row_sums = np.sum(confusion_matrix, axis=1)
        row_sums_adjusted = np.array([1 if val == 0 else val for val in row_sums])
        result = diag/row_sums_adjusted
        
    return result

def accuracy(y_true, y_pred):

    #get the num of rows in y_pred
    m = y_pred.shape[0]

    #difference of predicted vs true 
    dif = y_true - y_pred
    
    '''create an array, normalized_dif, of 0 and 1's where 0 represent correct predictions 
    and 1's are incorrect predictions'''
    normalized_dif = np.array([0 if val == 0 else 1 for val in dif])  
    
    #return 1 - (number of difference over total number of samples)
    return 1 - np.sum(normalized_dif) / m

def f1(y_true, y_pred, LABELS):
    p = precision(y_true, y_pred, 'macro', LABELS)
    r = recall(y_true, y_pred, 'macro', LABELS)
    return (2 * p * r)/(p + r)

def main():

    X_train, y_train_true = load_data_file(TRAIN_FILE)
    X_dev, y_dev_true = load_data_file(DEV_FILE)

    bayes_pipeline = build_naive_bayes()
    logistic_pipeline = build_logistic_regr()
    svm_pipeline = build_svm_pipeline()
    your_pipeline = build_own_pipeline()

    averages = ["micro", "macro"]
    n = len(averages)
    best_metrics = [[0, ""] for _ in range(2*n + 1)]

    for name, pipeline in (["Naive Bayes", bayes_pipeline], ["Logistic Regression", logistic_pipeline],
                          ["SVM", svm_pipeline], ["Your Pipeline", your_pipeline] 
                          ):

        if pipeline is not None:

            print("{}\n".format(name.upper()) + "="*50)
            
            #fit your pipeline with given training data
            pipeline.fit(X_train, y_train_true)
            
            #use your fitted pipeline to predict with dev data
            y_dev_pred = pipeline.predict(X_dev)
 
            for i, average in enumerate(averages):
        
                #calculate precision of true dev data with predicted dev data
                precisionCalc = precision(y_dev_true, y_dev_pred, average, LABELS)

                #calculate recall of true dev data with predicted dev data
                recallCalc = recall(y_dev_true, y_dev_pred, average, LABELS)

                #calculate accuracy of true dev data with predicted dev data
                accuracyCalc = accuracy(y_dev_true, y_dev_pred)

                #refer to format of best metrics to properly update the precision
                if precisionCalc > best_metrics[i][0]:
                    best_metrics[i][0] = precisionCalc
                    best_metrics[i][1] = name

                #refer to format of best metrics to properly update the precision
                if recallCalc > best_metrics[i+n][0]:
                    best_metrics[i+n][0] = recallCalc
                    best_metrics[i+n][1] = name

                #refer to format of best metrics to properly update the precision
                if accuracyCalc > best_metrics[2*n][0]:
                    best_metrics[2*n][0] = accuracyCalc
                    best_metrics[2*n][1] = name

                print("{} PRECISION: {}".format(average.upper(), precisionCalc))
                print("{} RECALL: {}".format(average.upper(), recallCalc))

            print("ACCURACY: {}".format(accuracyCalc))

            print("\n" + "="*50)

    print("BEST METRICS\n" + "="*50)
    for i in range(n):

        print("BEST {} PRECISION: {} {}".format(
                averages[i].upper(),
                best_metrics[i][0],
                best_metrics[i][1]
                ))

        print("BEST {} RECALL: {} {}".format(
                averages[i].upper(),
                best_metrics[i + n][0],
                best_metrics[i + n][1]
                ))

    print("BEST ACCURACY: {} {}".format(
            best_metrics[2*n][0],
            best_metrics[2*n][1]
            ))

    print("\n" + "="*50)

    print("GRID SEARCH\n" + "="*50)

    param_grid = {'clf__activation': ['logistic', 'tanh'], 'clf__alpha': [1, 0.5], 'tfidf__use_idf': [True, False],
                  'vect__max_df': [0.5, 0.75, 1], 'vect__ngram_range': [(1,2), (1, 1)]}

    search = GridSearchCV(your_pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    #fit the search GridSearchCV with trainingdata
    results = search.fit(X_train, y_train_true)

    #predict the search GridSearchCV with dev data
    y_dev_pred = results.predict(X_dev)

    #calculate precision of true dev data with predicted dev data
    precisionCalc = precision(y_dev_true, y_dev_pred, "micro", LABELS)

    #calculate recall of true dev data with predicted dev data
    recallCalc = recall(y_dev_true, y_dev_pred, "micro", LABELS)

    print("Estimator: ", results.best_estimator_)
    print("Best Score: ", results.best_score_)
    print("Best Params: ", results.best_params_)

    #Fitting search GridSearchCv with training and dev data
    search.fit(X_train + X_dev, y_train_true + y_dev_true)

    #call output_predictions on your search classifier
    output_predictions(search)
    
#The following code allows you to test if your precision and recall functions are working properly

def test():

    #Test precision and recall

    labels = ["blue","red","yellow",]

    true1 = ["blue", "red", "blue", "blue", "blue", "blue", "yellow"]
    pred1 = ["blue", "red", "yellow", "yellow", "red", "red", "red"]

    for (correct_precision, correct_recall, averaging) in [
        [0.4166666666666667, 0.39999999999999997, "macro"],
        [0.2857142857142857, 0.2857142857142857, "micro"]
        ]:

        our_recall = recall(true1, pred1, labels=labels, average=averaging)
        our_precision = precision(true1, pred1, labels=labels, average=averaging)

        print("\nAveraging: {}\n============".format(averaging))

        print("Recall\n-------")
        print("Correct: ", correct_recall)
        print("Ours: ", our_recall)
        print("")

        print("Precision\n---------")
        print("Correct: ", correct_precision)
        print("Ours: ", our_precision)
        print("")

        if correct_recall == our_recall and correct_precision == our_precision:

            print("All good!")

        else:

            print("Hmm, check implementation.")

main()

test()
