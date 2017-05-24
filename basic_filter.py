'''     
>>> print res
[4, 16, 36, 64]
'''

from pyspark import SparkContext

def even_squares(num):
    return num.filter(lambda x: x % 2 == 0).map(lambda x: x * x)
    

if __name__ == "__main__":
    
    sc = SparkContext('local', 'word_count')
    nums = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8])
    res = sorted(even_squares(nums).collect())
    print res
    