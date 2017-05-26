import os
import numpy as np
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

def parse_doc(values):
    return LabeledPoint(values[0], values[1:])

def main():
    
    #parameters
    num_features = 400  #vocabulary size
        
    #load data    
    print "loading 20 newsgroups dataset..."
    categories = ['rec.autos','rec.sport.hockey','comp.graphics','sci.space']    
    tic = time()
    dataset = fetch_20newsgroups(shuffle=True, random_state=0, categories=categories, remove=('headers','footers','quotes'))
    train_corpus = dataset.data  # a list of 11314 documents / entries
    train_labels = dataset.target 
    toc = time()
    print "elapsed time: %.4f sec" %(toc - tic)    
    
    #tf-idf vectorizer
    tfidf = TfidfVectorizer(max_df=0.5, max_features=num_features, \
                            min_df=2, stop_words='english', use_idf=True)
    X_tfidf = tfidf.fit_transform(train_corpus).toarray()
        
    #append document labels
    train_labels = train_labels.reshape(-1,1)
    X_all = np.hstack([train_labels, X_tfidf])

    #distribute the data    
    sc = SparkContext('local', 'log_reg')    
    rdd = sc.parallelize(X_all)    
    labeled_corpus = rdd.map(parse_doc)
    train_RDD, test_RDD = labeled_corpus.randomSplit([8, 2], seed=0)

    #distributed logistic regression
    print "training logistic regression..."
    model = LogisticRegressionWithLBFGS.train(train_RDD, regParam=1, regType='l1', numClasses=len(categories))

    #evaluated the model on test data
    labels_and_preds = test_RDD.map(lambda p: (p.label, model.predict(p.features)))    
    test_err = labels_and_preds.filter(lambda (v, p): v != p).count() / float(test_RDD.count())
    print "log-reg test error: ", test_err
    
    #model.save(sc, './log_reg_lbfgs_model')


if __name__ == "__main__":
    main()