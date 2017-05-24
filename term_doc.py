'''      
>>> tokens.first()
['It', 'is', 'the', 'east,', 'and', 'Juliet', 'is', 'the', 'sun.']

>>> local_vocab_map
{'and': 0, 'A': 1, 'fit': 14, 'for': 13, 'of': 3, 'is': 4, 'gods.': 7, 'It': 11,\
'Brevity': 10, 'soul': 12, 'sun.': 8, 'dish': 2, 'east,': 9, 'the': 5, 'wit.': 6, 'Juliet': 15}

>>> for doc in term_document_matrix.collect():
        print doc
(16,[0,4,5,8,9,11,15],[1.0,2.0,2.0,1.0,1.0,1.0,1.0])
(16,[1,2,5,7,13,14],[1.0,1.0,1.0,1.0,1.0,1.0])
(16,[3,4,5,6,10,12],[1.0,1.0,1.0,1.0,1.0,1.0])

'''

from pyspark.mllib.linalg import SparseVector
from collections import Counter

from pyspark import SparkContext

if __name__ == "__main__":
    
    sc = SparkContext('local', 'term_doc')
    corpus = sc.parallelize([
    "It is the east, and Juliet is the sun.",
    "A dish fit for the gods.",
    "Brevity is the soul of wit."])
        
    tokens = corpus.map(lambda raw_text: raw_text.split()).cache()   
    local_vocab_map = tokens.flatMap(lambda token: token).distinct().zipWithIndex().collectAsMap()
    
    vocab_map = sc.broadcast(local_vocab_map)
    vocab_size = sc.broadcast(len(local_vocab_map))
    
    term_document_matrix = tokens \
                         .map(Counter) \
                         .map(lambda counts: {vocab_map.value[token]: float(counts[token]) for token in counts}) \
                         .map(lambda index_counts: SparseVector(vocab_size.value, index_counts))

    for doc in term_document_matrix.collect():
        print doc
                