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

 
### Dependencies

PySpark 2.1.1  
Python 2.7

