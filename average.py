'''     
>>> sum_count
(55, 10)
>>> average
5.5
'''

from pyspark import SparkContext

if __name__ == "__main__":
    
    sc = SparkContext('local', 'word_count')
    nums = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sum_count = nums.map(lambda x: (x, 1)).fold((0,0), (lambda x, y: (x[0]+y[0], x[1]+y[1])))
    average = sum_count[0] / float(sum_count[1])    
    print average
    