from Recommender import Recommender

r = Recommender()
r.parseMovieDB("db/movies.dat", "db/user_ratedmovies-timestamps.dat")

r.averagePrediction(0)