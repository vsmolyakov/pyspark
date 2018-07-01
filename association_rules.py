from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import Row

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.ml.fpm import FPGrowth 

if __name__ == "__main__":

    sc = SparkContext('local', 'arules')
    sqlContext = SQLContext(sc)
    
    spark = SparkSession\
        .builder\
        .appName("arules")\
        .getOrCreate()
    
    #dataset = sc.textFile("./data/retail.txt")
    df = spark.createDataFrame([
        (0, [1, 2, 5]),
        (1, [1, 2, 3, 5]),
        (2, [1, 2])
    ], ["id", "items"])

    fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
    model = fpGrowth.fit(df)

    #display frequent itemsets
    model.freqItemsets.show()

    #display generated association rules
    model.associationRules.show()

    #apply transform
    model.transform(df).show()

