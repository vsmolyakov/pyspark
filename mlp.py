
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, IntegerType


if __name__ == "__main__":
    
    sc = SparkContext('local', 'mlp')
    sqlContext = SQLContext(sc)
    
    spark = SparkSession\
        .builder\
        .appName("MLPClassifier")\
        .getOrCreate()
    
    #read in csv as dataframe
    dataset = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('./data/titanic.csv')   
    dataset = dataset.drop('PassengerId','Name','Ticket','Cabin')       
    
    #set column types
    dataset = dataset.withColumn("Survived", dataset["Survived"].cast(IntegerType()))
    dataset = dataset.withColumn("Pclass", dataset["Pclass"].cast(IntegerType()))    
    dataset = dataset.withColumn("Age", dataset["Age"].cast(DoubleType()))
    dataset = dataset.withColumn("SibSp", dataset["SibSp"].cast(IntegerType()))
    dataset = dataset.withColumn("Parch", dataset["Parch"].cast(IntegerType()))
    dataset = dataset.withColumn("Fare", dataset["Fare"].cast(DoubleType()))
        
    #fill NaN
    avg_age = round(dataset.groupBy().avg("age").collect()[0][0],2)
    dataset = dataset.na.fill({'Age': avg_age})
    dataset = dataset.na.drop()

    #map categorical data
    indexer = StringIndexer(inputCol="Sex", outputCol="SexInd")
    dataset = indexer.fit(dataset).transform(dataset)
    
    indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedInd")
    dataset = indexer.fit(dataset).transform(dataset)
    
    #assemble features
    assembler = VectorAssembler(
        inputCols=["Age","Pclass","SexInd","SibSp","Parch","Fare","EmbarkedInd"],
        outputCol="features")

    dataset = assembler.transform(dataset)
        
    (trainingData, testData) = dataset.randomSplit([0.8, 0.2])
    
    #MLP
    layers = [7, 8, 4, 2]  #input: 7 features; output: 2 classes
    mlp = MultilayerPerceptronClassifier(maxIter=100, layers=layers, labelCol="Survived", featuresCol="features", blockSize=128, seed=0)
    
    model = mlp.fit(trainingData)    
    result = model.transform(testData)
    
    prediction_label = result.select("prediction", "Survived")
    evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
    print "MLP test accuracy: " + str(evaluator.evaluate(prediction_label))
    
    