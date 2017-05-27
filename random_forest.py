import numpy as np
import pandas as pd

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import Row

from sklearn.preprocessing import LabelEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


if __name__ == "__main__":
    
    sc = SparkContext('local', 'dataframe')
    sqlContext = SQLContext(sc)
    
    spark = SparkSession\
        .builder\
        .appName("RandomForestClassifier")\
        .getOrCreate()
        
    #dataset = sc.textFile("./data/titanic.csv", 1)
    #dataset = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('./data/titanic.csv')
    dataset = pd.read_csv("./data/titanic.csv")    
    dataset = dataset.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
    
    #map categorical data
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(dataset[col])
            dataset[col] = lbl.transform(dataset[col])
    
    #fill NaN
    median_age = dataset['Age'].dropna().median()
    dataset['Age'] = dataset['Age'].fillna(median_age)
    #dataset.isnull().sum()
    
    rdd_data = sc.parallelize(dataset.values)
    rdd_row = rdd_data.map(lambda p: Row(survived=int(p[0]),pclass=int(p[1]),sex=int(p[2]),age=float(p[3]),sibsp=int(p[4]),parch=int(p[5]),fare=float(p[6]),embarked=int(p[7])))
    titanic = spark.createDataFrame(rdd_row)
    
    assembler = VectorAssembler(
        inputCols=["age","embarked","fare","parch","pclass","sex","sibsp"],
        outputCol="features")
    
    titanic = assembler.transform(titanic)        
    (trainingData, testData) = titanic.randomSplit([0.8, 0.2])
    
    #random forest classifier
    rfc = RandomForestClassifier(numTrees=100, maxDepth=6, labelCol="survived", featuresCol="features", seed=0)
    model = rfc.fit(trainingData)
    
    #feature importances
    model.featureImportances
    
    #predictions
    predictions = model.transform(testData)
    predictions.select("survived", "probability", "prediction").show(truncate=False)

    #compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="survived", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print "RFC accuracy = %2.4f" % accuracy
    
            