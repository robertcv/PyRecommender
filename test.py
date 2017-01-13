from Recommender import Recommender
from Predictors import RandomPredictor
rp = RandomPredictor(1, 5)
r = Recommender(rp)
r.parseMovieDB("db/movies.dat", "db/user_ratedmovies-timestamps.dat")
print(r.userID)
print(r.userRatings)