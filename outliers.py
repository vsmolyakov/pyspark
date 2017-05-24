'''      
>>> print output
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
'''

from pyspark import SparkContext

def remove_outliers(nums):
    
    stats = nums.stats()
    stddev = stats.stdev()
    return nums.filter(lambda x: abs(x-stats.mean()) < 3 * stddev)

if __name__ == "__main__":
    
    sc = SparkContext('local', 'outliers')    
    nums = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000])
    output = sorted(remove_outliers(nums).collect())
    print output
    
    