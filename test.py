from Recommender import Recommender

r = Recommender()
r.parseMovieDB("db/movies.dat", "db/user_ratedmovies-timestamps.dat")
# print('____________________________________________________________')
# r.averagePrediction(100)
# print('____________________________________________________________')
# r.averagePrediction(100, user='1')
# print('____________________________________________________________')
# r.ItemBasedPredictionFit(10, 0.3)
# print('____________________________________________________________')
# r.ItemBasedRecommendUser('75', n=10)
# print('____________________________________________________________')
# r.ItemBasedBest(n=5)
# print('____________________________________________________________')
# r.UserBasedPredictionFit(10, 0.3)
# print('____________________________________________________________')
# r.UserBasedBest(n=5)
# print('____________________________________________________________')
# r.UserBasedRecommendUser('6393', n=10) #6393, 54767
# print('____________________________________________________________')
# r.SlopeOnePredictionFit()
# print('____________________________________________________________')
# r.SlopeOneRecommendUser('6393', n=10)
# print('____________________________________________________________')
# r.Evaluet(10, 0.3)
# print('____________________________________________________________')
# r.HybridPredictionFit(10, 0.3)
# print('____________________________________________________________')
# r.HybridRecommendUser('6393', n=10)
# print('____________________________________________________________')
r.MatrixFactorizationFit()
# print('____________________________________________________________')
# r.MatrixFactorizationRecommendUser('6393')
# print('____________________________________________________________')
r.MatrikFactorizationGraph()
# print('____________________________________________________________')
# r.parseTageDB("db/movie_tags.dat", "db/tags.dat")
# print('____________________________________________________________')
# r.NaiveBayes('6393', 2, n=20)
# print('____________________________________________________________')

