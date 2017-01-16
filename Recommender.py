import numpy as np
from math import sqrt
from itertools import combinations
from collections import defaultdict
import time

class Recommender(object):
    def __init__(self):
        self.movieID = None
        self.movieTitle = []
        self.movieDict = {}
        self.userID = None
        self.userRatings = None
        self.recommendation = None
        self.X = None
        self.newWidth = None
        self.newMovies = None
        self.i = 0


    def parseMovieDB(self, movie_file, ratedmovies_file):
        start = time.time()
        movie = open(movie_file, 'rt', encoding='utf-8')
        movieID = []
        movieTitle = []

        first_line = movie.readline() #header line
        i = 0
        for line in movie:
            ln = line.strip().split('\t')
            movieID.append(ln[0])
            movieTitle.append(ln[1]) #save title
            self.movieDict[ln[0]] = i #save position of movie
            i += 1

        self.width = len(movieID) #set the width of the rating matix

        #code for finding the number of users
        """
        ratedmovies = open(ratedmovies_file, 'rt', encoding='utf-8')
        userSet = set()
        first_line = ratedmovies.readline()
        for line in ratedmovies:
            ln = line.strip().split('\t')
            userSet.add(ln[0])
        """
        self.height = 2113 #len(userSet)  set the height of the rating matix
        #for speed its hardcoded

        ratedmovies = open(ratedmovies_file, 'rt', encoding='utf-8')
        userID = [] #save user ids
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
                userID.append(curentU) #save the old
                self.userRatings[j][self.movieDict[ln[1]]] = ln[2] #save rating
                curentU = ln[0] #save new user
                j += 1

        userID.append(curentU) #save the last user

        self.userID = np.array(userID)
        self.movieID = np.array(movieID)
        self.movieTitle = np.array(movieTitle)
        print("Time to load data: {0:.2f}s".format(time.time() - start))

    def averagePrediction(self, b, n=20, user=None):
        start = time.time()
        #recommends movies with best avrage ratings
        ratingsSum = np.nansum(self.userRatings, axis=0) #add all ratings of movies
        ratingsNum = np.sum(~np.isnan(self.userRatings), axis=0) #count the number of given ratings
        globalAvg = np.sum(ratingsSum)/np.sum(ratingsNum) #calculate global avrage

        nonzero = ratingsNum!=0 #finde all movies wich have atlest one rating
        self.recommendation = np.zeros(self.width)
        self.recommendation[nonzero] = (ratingsSum[nonzero] + b * globalAvg) / (ratingsNum[nonzero] + b) #calculate movie avrage
        print("Time to calculate movie average: {0:.2f}s".format(time.time() - start))

        dtype = [('title', np.dtype((str, 100))), ('rating', float)]  # define numpy data type
        if user is not None: #check if user was given
            userIndex = np.where(self.userID == user)[0][0] #finde user index
            seenMovies = ~np.isnan(self.userRatings[userIndex, :]) #finde seen movies
            rec = np.array(list(zip(self.movieTitle[seenMovies], self.recommendation[seenMovies])), dtype=dtype) #print average of those seen movies
        else:
            rec = np.array(list(zip(self.movieTitle, self.recommendation)), dtype=dtype)  # combine movie rating with title

        rec = np.sort(rec, order='rating')[-n:][::-1] #sort, take last n, reverse
        for r in rec:
            print(r)

    def similarityFun(self, i1, i2, K):
        notNan = ~np.isnan(i1) & ~np.isnan(i2) #finde where both are numbers
        i1 = i1[notNan] #remember, for speed
        i2 = i2[notNan] #remember, for speed
        cov = np.dot(i1,i2)
        var1 = np.dot(i1,i1)
        var2 = np.dot(i2,i2)
        return cov/(sqrt(var1)*sqrt(var2)+K)

    def ItemBasedPredictionFit(self, K):
        start = time.time()
        self.usableRatings = np.sum(~np.isnan(self.userRatings), axis=0) > 300 #get movies with atleast m ratings
        self.X = self.userRatings[:,self.usableRatings] #save them

        self.userAve = np.array([np.nanmean(self.X, axis=1)]) #calculate mean for every user
        self.userAve = self.userAve.T #transpose for subtracting
        self.X = self.X - self.userAve #subtract mean of user from their ratings
        self.newWidth = self.X.shape[1] #save new width
        self.newMovies = self.movieID[self.usableRatings] #save reduced movieID table

        self.recommendation = np.zeros((self.newWidth,self.newWidth)) #crate ampty array
        for i1, i2 in combinations(range(self.newWidth), 2): #get all combinations of movies
            s = self.similarityFun(self.X[:,i1], self.X[:,i2], K) #calculate similarity
            self.recommendation[i1, i2] = s #save it
            self.recommendation[i2, i1] = s

        print("Time to calculate similarity: {0:.2f}s".format(time.time() - start))

    def ItemBasedRecommend(self, user=None, n=20, recSeen=True):
        if user is not None:
            self.ItemBasedRecommendUser(user, n, recSeen) #seperate user recomend
        else:
            self.ItemBasedRecommendBest(n) #from best similarity

    def ItemBasedRecommendUser(self, user, n, recSeen):
        start = time.time()
        userIndex = np.where(self.userID == user)[0][0] #finde user index
        userRatings = np.nan_to_num(self.userRatings[userIndex, :]) #transform unseen movies to 0 rating
        maxRatings = userRatings >= np.sort(userRatings)[-n] #sort all ratings and get those with th higewst rating
        aboveAverage = userRatings >= self.userAve[userIndex] #get those above avergae
        usableMovies = self.usableRatings & maxRatings & aboveAverage #combine
        movies = self.movieID[usableMovies] #highest rated movies form user

        similarMovies = defaultdict(int) #for saving similar movies and their similarity
        for movieId in movies:
            movieIndex = np.where(self.newMovies == movieId)[0][0] #get movie index
            max = np.max(self.recommendation[movieIndex, :]) #get max similarity
            similarMovieIndex = np.where(self.recommendation[movieIndex, :] == max)[0][0] #finde the index
            similarMovie = self.newMovies[similarMovieIndex] #finde movie id
            if similarMovies[similarMovie] < max: #save if not allredy
                similarMovies[similarMovie] = max

        print("Time to finde similar movies: {0:.2f}s".format(time.time() - start))

        seenMovies = self.movieTitle[~np.isnan(self.userRatings[userIndex, :])] #get ids of seen movies
        result = []
        for key, value in similarMovies.items():
            if recSeen or key not in seenMovies:
                result.append((self.movieTitle[self.movieDict[key]], value)) #save similar movies and similarity

        result.sort(key=lambda tup: -tup[1]) #sort them
        if len(result) > n:
            for r in result[:n]:
                print(r)
        else:
            for r in result:
                print(r)

    def ItemBasedRecommendBest(self, n):
        start = time.time()
        result = []
        for i1, i2 in combinations(range(self.newWidth), 2):
            result.append((i1, i2, self.recommendation[i1, i2])) #save all similarities

        result = sorted(result, key=lambda tup: -tup[2])[:n] #sort them an get the best
        print("Time to finde similar movies: {0:.2f}s".format(time.time() - start))

        for r in result: #print the best
            i1 = self.newMovies[r[0]] #get index
            m1 = self.movieTitle[self.movieDict[i1]] #get title
            i2 = self.newMovies[r[1]] #get index
            m2 = self.movieTitle[self.movieDict[i2]] #get title
            print((m1, m2, r[2]))
