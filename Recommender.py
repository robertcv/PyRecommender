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
        self.usableRatings = None
        self.X = None
        self.newWidth = None
        self.newMovies = None


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
        self.height = 2114 # 2113 #len(userSet)  set the height of the rating matix
        #for speed its hardcoded

        ratedmovies = open(ratedmovies_file, 'rt', encoding='utf-8')
        userID = [] #save user ids
        self.userRatings = np.full((self.height, self.width), np.nan) #crate an empty matrix

        first_line = ratedmovies.readline() #header line

        ln = ratedmovies.readline().split('\t') #reade first line
        curentU = ln[0] #remember curent user
        self.userRatings[0, self.movieDict[ln[1]]] = ln[2] #set first rating

        j = 0
        for line in ratedmovies:
            ln = line.strip().split('\t')
            if ln[0] == curentU: #if the user is the same just save te rating for the movie
                self.userRatings[j, self.movieDict[ln[1]]] = ln[2]
            else: #if its a new user
                userID.append(curentU) #save the old
                curentU = ln[0]  # save new user
                j += 1
                self.userRatings[j, self.movieDict[ln[1]]] = ln[2] #save rating

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

    def ItemBasedPredictionFit(self, K, threshold):
        start = time.time()
        self.usableRatings = np.sum(~np.isnan(self.userRatings), axis=0) > 300 #get movies with atleast m ratings
        self.X = np.copy(self.userRatings[:, self.usableRatings]) #save them

        self.userAve = np.array([np.nanmean(self.X, axis=1)]) #calculate mean for every user
        self.userAve = self.userAve.T #transpose for subtracting
        self.X = self.X - self.userAve #subtract mean of user from their ratings
        self.newWidth = self.X.shape[1] #save new width
        self.newMovies = self.movieID[self.usableRatings] #save reduced movieID table

        self.recommendation = np.zeros((self.newWidth,self.newWidth)) #crate ampty array
        for i1, i2 in combinations(range(self.newWidth), 2): #get all combinations of movies
            s = self.similarityFun(self.X[:, i1], self.X[:, i2], K) #calculate similarity
            if s < threshold: s = 0
            self.recommendation[i1, i2] = s #save it
            self.recommendation[i2, i1] = s

        print("Time to calculate similarity: {0:.2f}s".format(time.time() - start))

    def ItemBasedRecommendUser(self, user, n=20, recSeen=True):
        start = time.time()
        userIndex = np.where(self.userID == user)[0][0] #finde user index
        notRatedMovies = np.argwhere(np.isnan(self.X[userIndex, :])).ravel()
        ratedMovies = ~np.isnan(self.X[userIndex, :])
        userRatingValue = self.X[userIndex, ratedMovies]

        moviesRecommend = np.zeros(self.newWidth)
        for i in notRatedMovies:
            similarity = self.recommendation[i, ratedMovies]
            similaritySum = np.sum(similarity)
            if similaritySum == 0:
                rating = 0
            else:
                rating = np.dot(userRatingValue, similarity) / similaritySum
            moviesRecommend[i] = self.userAve[userIndex] + rating

        print("Time to finde similar movies: {0:.2f}s".format(time.time() - start))

        seenMovies = np.isnan(self.X[userIndex, :])
        result = []
        for i in range(self.newWidth):
            if recSeen or seenMovies[i]:
                result.append((self.movieTitle[self.movieDict[self.newMovies[i]]], moviesRecommend[i]))

        result.sort(key=lambda tup: -tup[1]) #sort them
        if len(result) > n:
            for r in result[:n]:
                print("{} - {:.2f}".format(r[0], r[1]))
        else:
            for r in result:
                print("{} - {:.2f}".format(r[0], r[1]))

    def ItemBasedBest(self, n=20):
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

    def UserBasedPredictionFit(self, K):
        start = time.time()
        self.usableRatings = np.sum(~np.isnan(self.userRatings), axis=1) > 500  # get users with atleast m ratings
        self.X = self.userRatings[self.usableRatings, :]  # save them

        self.userAve = np.array([np.nanmean(self.X, axis=1)]) #calculate mean for every user
        self.userAve = self.userAve.T #transpose for subtracting
        self.X = self.X - self.userAve #subtract mean of user from their ratings

        self.newHeight = self.X.shape[0]  # save new width
        self.newUser = self.userID[self.usableRatings]  # save reduced movieID table
        self.recommendation = np.zeros((self.newHeight,self.newHeight)) #crate ampty array

        for u1, u2 in combinations(range(self.newHeight), 2): #get all combinations of movies
            s = self.similarityFun(self.X[u1, :], self.X[u2, :], K) #calculate similarity
            self.recommendation[u1, u2] = s #save it
            self.recommendation[u2, u1] = s

        print("Time to calculate similarity: {0:.2f}s".format(time.time() - start))

    def UserBasedRecommend(self, user=None, n=20, recSeen=True):
        if user is not None:
            self.UserBasedRecommendUser(user, n, recSeen) #seperate user recomend
        else:
            self.UserBasedRecommendBest(n) #from best similarity

    def UserBasedRecommendUser(self, user, n, recSeen):
        start = time.time()
        newUserIndex = np.where(self.newUser == user)[0][0] #finde user index
        maxSimilarUser = np.nanmax(self.recommendation[newUserIndex, :])
        similarUsers = self.recommendation[newUserIndex, :]


        print("Time to finde similar movies: {0:.2f}s".format(time.time() - start))



    def UserBasedRecommendBest(self, n):
        start = time.time()
        result = []
        for u1, u2 in combinations(range(self.newHeight), 2):
            result.append((u1, u2, self.recommendation[u1, u2])) #save all similarities

        result = sorted(result, key=lambda tup: -tup[2])[:n] #sort them an get the best
        print("Time to finde similar movies: {0:.2f}s".format(time.time() - start))

        for r in result: #print the best
            u1 = self.newUser[r[0]] #get id
            u2 = self.newUser[r[1]] #get id
            print((u1, u2, r[2]))