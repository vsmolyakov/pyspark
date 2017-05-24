'''      
>>> lines.collect()
[u'ATATCCCCGGGAT', u'ATCGATCGATATG']

>>> rdd.collect()
[(u'A', 3), (u'C', 4), (u'T', 3), (u'G', 3), (u'A', 4), (u'C', 2), (u'T', 4), (u'G', 3)]

>>> cnt.collect()
[(u'A', 7), (u'C', 6), (u'T', 7), (u'G', 6)]
'''

def mapper(seq):
    freq = dict()
    for x in list(seq):
        if x in freq:
            freq[x] += 1
        else:
            freq[x] = 1
    
    kv = [(x, freq[x]) for x in freq.keys()]    
    return kv


from pyspark import SparkContext

if __name__ == "__main__":
    
    sc = SparkContext('local', 'mapper')
    lines = sc.textFile("./data/dna_seq.txt", 1)    

    rdd = lines.flatMap(mapper)
    cnt = rdd.reduceByKey(lambda x, y: x + y)
    print cnt.collect()
    