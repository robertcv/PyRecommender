import numpy as np

class Recommender(object):

    def parseMovieDB(self, movie_file, ratedmovies_file):
        movie = open(movie_file, 'rt', encoding='utf-8')
        self.movieTitle = []
        self.movieID = []
        self.movieDict = {}

        first_line = movie.readline() #header line
        i = 0
        for line in movie:
            ln = line.strip().split('\t')
            self.movieID.append(ln[0])
            self.movieTitle.append(ln[1]) #save title
            self.movieDict[ln[0]] = i #save position of movie
            i += 1

        self.width = len(self.movieID) #set the width of the rating matix

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
        self.height = 2113 #len(userSet)  set the height of the rating matix

        ratedmovies = open(ratedmovies_file, 'rt', encoding='utf-8')
        self.userID = [] #save user ids
        self.userRatings = np.full((self.height, self.width), np.nan) #crate an empty matrix

        first_line = ratedmovies.readline() #header line

        ln = ratedmovies.readline().split('\t') #reade first line
        curentU = ln[0] #remember curent user
        self.userRatings[0][self.movieDict[ln[1]]] = ln[2] #set first rating

        j = 0
        for line in ratedmovies:
            ln = line.strip().split('\t')
            if ln[0] == curentU: #if the user is the same just save te rating for the movie
                self.userRatings[j][self.movieDict[ln[1]]] = ln[2]
            else: #if its a new user
                self.userID.append(curentU) #save the old
                self.userRatings[j][self.movieDict[ln[1]]] = ln[2] #save rating
                curentU = ln[0] #save new user
                j += 1

        self.userID.append(curentU) #save the last user

    def averagePrediction(self, b, n=20):
        #recommends movies with best avrage ratings
        ratingsSum = np.nansum(self.userRatings, axis=0)
        ratingsNum = np.sum(~np.isnan(self.userRatings), axis=0)
        globalAvg = np.sum(ratingsSum)/np.sum(ratingsNum)

        recommendation = (ratingsSum + b * globalAvg) / (ratingsNum + b)

        dtype = [('title', np.dtype((str,100))), ('rating', float)]
        rec = np.array(list(zip(self.movieTitle, recommendation)), dtype=dtype)
        rec.sort(order='rating')
        rec = rec[-n:]
        print(rec[::-1])