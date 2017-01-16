from Recommender import Recommender

r = Recommender()
r.parseMovieDB("db/movies.dat", "db/user_ratedmovies-timestamps.dat")

#r.averagePrediction(100)
#r.averagePrediction(100, user='75')
r.ItemBasedPredictionFit(10)
#r.ItemBasedRecommend(user='75', n=10, recSeen=False)
r.ItemBasedRecommend()