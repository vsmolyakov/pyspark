'''      
>>> lines.collect()
[u'hello world', u'hello pyspark', u'spark context', u'i like spark', u'hadoop rdd', u'text file', u'word count', u'', u'']

>>> words.collect()
[u'hello', u'world', u'hello', u'pyspark', u'spark', u'context', u'i', u'like', u'spark', u'hadoop', u'rdd', u'text', u'file', u'word', u'count', u'', u'']

>>> ones.take(2)
[(u'hello', 1), (u'world', 1)]

>>> counts.takeSample(1, 2)
[(u'spark', 2), (u'hello', 2)]
'''

from pyspark import SparkContext

if __name__ == "__main__":
    
    sc = SparkContext('local', 'word_count')
    lines = sc.textFile("./data/words.txt", 1)    

    words = lines.flatMap(lambda x: x.split(' '))    
    ones = words.map(lambda x: (x, 1))
    counts = ones.reduceByKey(lambda x, y: x + y)
    counts = counts.sortByKey(1)
    
    counts.saveAsTextFile("./data/word_counts.txt")