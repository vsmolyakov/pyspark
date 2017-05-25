'''      
>>> sum_cnt
(55, 10)
'''

from pyspark import SparkContext

if __name__ == "__main__":
    
    sc = SparkContext('local', 'aggregate')    
    nums = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    sum_cnt = nums.aggregate(
        (0,0), #initial value
        (lambda acc, value: (acc[0] + value, acc[1] + 1)), #combine value with acc
        (lambda acc1, acc2: (acc1[0]+acc2[0],acc1[1]+acc2[1])) #combine accumulators
    )
    
    print "mean: ", round(sum_cnt[0]/float(sum_cnt[1]),4)
    
    
    