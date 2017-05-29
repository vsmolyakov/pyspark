
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import Row

import re
import numpy as np
from time import time
from sklearn.datasets import fetch_20newsgroups

from pyspark.ml.feature import CountVectorizer, HashingTF, IDF
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.clustering import LDA

np.random.seed(0)

if __name__ == "__main__":

    sc = SparkContext('local', 'lda')
    sqlContext = SQLContext(sc)
    
    spark = SparkSession\
        .builder\
        .appName("LDA")\
        .getOrCreate()
    

    num_features = 8000  #vocabulary size
    num_topics = 20      #fixed for LDA

    print "loading 20 newsgroups dataset..."
    tic = time()
    dataset = fetch_20newsgroups(shuffle=True, random_state=0, remove=('headers','footers','quotes'))
    train_corpus = dataset.data  # a list of 11314 documents / entries
    toc = time()
    print "elapsed time: %.4f sec" %(toc - tic)    
    
    #distribute data
    corpus_rdd = sc.parallelize(train_corpus)
    corpus_rdd = corpus_rdd.map(lambda doc: re.sub(r"[^A-Za-z]", " ", doc))
    corpus_rdd = corpus_rdd.map(lambda doc: u"".join(doc).encode('utf-8').strip())
        
    rdd_row = corpus_rdd.map(lambda doc: Row(raw_corpus=str(doc)))
    newsgroups = spark.createDataFrame(rdd_row)
    
    tokenizer = Tokenizer(inputCol="raw_corpus", outputCol="tokens")
    newsgroups = tokenizer.transform(newsgroups)
    newsgroups = newsgroups.drop('raw_corpus')       

    stopwords = StopWordsRemover(inputCol="tokens", outputCol="tokens_filtered")
    newsgroups = stopwords.transform(newsgroups)
    newsgroups = newsgroups.drop('tokens')
    
    count_vec = CountVectorizer(inputCol="tokens_filtered", outputCol="tf_features", vocabSize=num_features, minDF=2.0)
    count_vec_model = count_vec.fit(newsgroups)
    vocab = count_vec_model.vocabulary
    newsgroups = count_vec_model.transform(newsgroups)
    newsgroups = newsgroups.drop('tokens_filtered')
    
    #hashingTF = HashingTF(inputCol="tokens_filtered", outputCol="tf_features", numFeatures=num_features)
    #newsgroups = hashingTF.transform(newsgroups)
    #newsgroups = newsgroups.drop('tokens_filtered')

    idf = IDF(inputCol="tf_features", outputCol="features")
    newsgroups = idf.fit(newsgroups).transform(newsgroups)
    newsgroups = newsgroups.drop('tf_features')
    
    lda = LDA(k=num_topics, featuresCol="features", seed=0)
    model = lda.fit(newsgroups)
 
    topics = model.describeTopics()
    topics.show()
    
    model.topicsMatrix()
    
    topics_rdd = topics.rdd

    topics_words = topics_rdd\
       .map(lambda row: row['termIndices'])\
       .map(lambda idx_list: [vocab[idx] for idx in idx_list])\
       .collect()

    for idx, topic in enumerate(topics_words):
        print "topic: ", idx
        print "----------"
        for word in topic:
            print word
        print "----------"
    
    
    





