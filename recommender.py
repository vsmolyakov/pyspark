
import os
import math
from time import time
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS

DATASET_PATH = '/data/vision/fisher/data1/MovieLens/'

def main():
    
    sc = SparkContext('local', 'als')    
    
    small_ratings_file = os.path.join(DATASET_PATH,'ml-latest-small','ratings.csv')
    small_ratings_raw_data = sc.textFile(small_ratings_file)
    small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

    #userId, movieId, rating, timestamp
    small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()
                            
    small_movies_file = os.path.join(DATASET_PATH, 'ml-latest-small', 'movies.csv')
    small_movies_raw_data = sc.textFile(small_movies_file)
    small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

    #movieId, title, genre
    small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()
    small_movies_titles = small_movies_data.map(lambda x: (int(x[0]),x[1]))


    #training, validation, test split
    training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0L)
    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
    
    #ALS parameters and model selection
    iterations = 10
    regularization_parameter = 0.1
    ranks = [4, 8, 12]
    tolerance = 0.02
    errors = []
    
    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    
    for rank in ranks:
        model = ALS.train(training_RDD, rank, seed=0, iterations=iterations, lambda_=regularization_parameter)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))   #((user, product), rating)
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions) #((user, product), rating1, rating2)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())   #sqrt[mean((rating1 - rating2)**2)]
        errors.append(error)
        
        print 'For rank %s the RMSE is %s' % (rank, error)
        if error < min_error:
            min_error = error
            best_rank = rank

    print 'The best model was trained with rank %s' % best_rank
    
    #new user ratings
    new_user_ID = 0
    #userId, movieId, rating
    new_user_ratings = [
     (0,260,9), # Star Wars (1977)
     (0,1,8), # Toy Story (1995)
     (0,16,7), # Casino (1995)
     (0,25,8), # Leaving Las Vegas (1995)
     (0,32,9), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,4), # Flintstones, The (1994)
     (0,379,3), # Timecop (1994)
     (0,296,7), # Pulp Fiction (1994)
     (0,858,10) , # Godfather, The (1972)
     (0,50,8) # Usual Suspects, The (1995)
    ]
    new_user_ratings_RDD = sc.parallelize(new_user_ratings)
    print 'New user ratings: %s' % new_user_ratings_RDD.take(10)
    
    #re-train the model
    small_data_with_new_ratings_RDD = small_ratings_data.union(new_user_ratings_RDD)
    t0 = time()
    new_ratings_model = ALS.train(small_data_with_new_ratings_RDD, best_rank, seed=0, iterations=iterations, lambda_=regularization_parameter)
    tt = time() - t0
    print "New model trained in %s seconds" % round(tt,3)    
    
    #getting top recommendations
    new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just movie IDs

    #movieId, title, genre
    new_user_unrated_movies_RDD = (small_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))
    new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
    
    #movieId, rating
    new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
    new_user_recommendations_rating_title_RDD = new_user_recommendations_rating_RDD.join(small_movies_titles)
    new_user_recommendations_rating_title_RDD = new_user_recommendations_rating_title_RDD.map(lambda r: (r[1][1], r[1][0]))
        
    top_movies = new_user_recommendations_rating_title_RDD.takeOrdered(25, key=lambda x: -x[1])    
    print "top recommended movies:\n %s" % '\n'.join(map(str, top_movies))
     
    
    
    
    
    
    
    

if __name__ == "__main__":    
    main()