import numpy as np

class Recommender(object):
    def __init__(self, predictor=None):
        self.predictor = predictor


    def parseMovieDB(self, movie_file, ratedmovies_file):
        movie = open(movie_file, 'rt', encoding='utf-8')
        self.movieTitle = {}
        self.movieID = []
        self.movieDict = {}

        first_line = movie.readline()
        i = 0
        for line in movie:
            ln = line.strip().split('\t')
            self.movieID.append(ln[0])
            self.movieTitle[ln[0]] = ln[1]
            self.movieDict[ln[0]] = i
            i += 1

        self.width = len(self.movieID)

        #code for finding the number of users
        """
        ratedmovies = open(ratedmovies_file, 'rt', encoding='utf-8')
        userSet = set()
        first_line = ratedmovies.readline()
        for line in ratedmovies:
            ln = line.strip().split('\t')
            userSet.add(ln[0])
        """
        #for speed its hardcoded
        self.height = 2113 #len(userSet)

        ratedmovies = open(ratedmovies_file, 'rt', encoding='utf-8')
        self.userID = []
        self.userRatings = np.full((self.height, self.width), np.nan)

        first_line = ratedmovies.readline()
        ln = ratedmovies.readline().split('\t')

        curentU = ln[0]
        self.userRatings[0][self.movieDict[ln[1]]] = ln[2]

        j = 0
        for line in ratedmovies:
            ln = line.strip().split('\t')
            if ln[0] == curentU:
                self.userRatings[j][self.movieDict[ln[1]]] = ln[2]
            else:
                self.userID.append(curentU)
                self.userRatings[j][self.movieDict[ln[1]]] = ln[2]
                curentU = ln[0]
                j += 1

        self.userID.append(curentU)

    def learn(self):
        self.predictor.fit(self.userRatings)

    '''
        def recommend(self, user, n=10, rec_seen=True):
        user_id = self.users.index(user)
        rec_tup = np.array(list(zip(self.items, self.predictor.predict(user_id))))
        rec_bool = np.logical_not(np.array(self.userRatings[user_id], dtype=bool)) | rec_seen
        rec = rec_tup[rec_bool]
        return rec[rec[:,1].argsort()][-n:]

    '''