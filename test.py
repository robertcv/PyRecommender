from Recommender import Recommender

r = Recommender()
r.parseMovieDB("db/movies.dat", "db/user_ratedmovies-timestamps.dat")


#r.averagePrediction(100)
#r.averagePrediction(100, user='75')

r.ItemBasedPredictionFit(10, 0.3)
r.ItemBasedRecommendUser('1', n=10)
#r.ItemBasedBest(n=5)