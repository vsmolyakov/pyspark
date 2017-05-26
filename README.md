# pyspark
pyspark

### Description

This repo is a collection of pySpark resources.

<p align="center">
<img src="https://github.com/vsmolyakov/pyspark/blob/master/figures/spark.png" />
</p>

References:  
*https://spark.apache.org/docs/latest/programming-guide.html#transformations*  
*https://spark.apache.org/docs/latest/ml-guide.html*

**ALS Recommender System**

Alternating Least Squares (ALS) matrix factorization recommender system was used to predict top movies for a new user given the ratings in the MovieLens dataset.

<p align="center">
<img src="https://github.com/vsmolyakov/pyspark/blob/master/figures/als.png" />
</p>

RMSE was computed for different ALS ranks in order to select the best model, which was then used to predict movie ratings and recommend highest rated movies to a new user.

References:  
*https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html*  

**Distributed Logistic Regression**

LBFGS Logistic Regression was used to classify the 20newsgroups dataset according to one of 20 topics. Each document in a corpus was converted to a tf-idf vector labelled by the corresponding topic for training. 

<p align="center">
<img src="https://github.com/vsmolyakov/pyspark/blob/master/figures/20newsgroups.png" />
</p>

A test accuracy was computed by predicting the topic label based on test tf-idf document vectors. The figure above shows a t-SNE visualization of the 20newsgroups corpus.

References:  
*https://spark.apache.org/docs/latest/mllib-linear-methods.html#logistic-regression*  



**Misc**

[RDD aggregation](https://github.com/vsmolyakov/pyspark/blob/master/aggregate.py), [RDD filter](https://github.com/vsmolyakov/pyspark/blob/master/basic_filter.py), [RDD mapper](https://github.com/vsmolyakov/pyspark/blob/master/mapper.py),     
[word count](https://github.com/vsmolyakov/pyspark/blob/master/word_count.py), [term document matrix](https://github.com/vsmolyakov/pyspark/blob/master/term_doc.py), [average](https://github.com/vsmolyakov/pyspark/blob/master/average.py), [outliers](https://github.com/vsmolyakov/pyspark/blob/master/outliers.py), [pi_est](https://github.com/vsmolyakov/pyspark/blob/master/pi_est.py)
 
### Dependencies

PySpark 2.1.1  
Python 2.7

