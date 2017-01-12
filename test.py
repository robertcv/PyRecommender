from Recommender import Recommender
from Predictors import RandomPredictor


rp = RandomPredictor(1, 5)
r = Recommender(rp)
r.parse_moviebase("moviebase2016.txt")
r.learn()
print(r.recommend("sasha", n=2))
print(r.recommend("sasha"))
print(r.recommend("sasha", rec_seen=False))