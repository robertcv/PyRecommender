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

        self.usableMovies = None
        self.usableUsers = None
        self.userAve = None
        self.X = None
        self.newWidth = None
        self.newHeight = None
        self.newUser = None
        self.newMovies = None

        self.itemSimilarity = None
        self.userSimilarity = None
        self.s1DifferenceNum = None
        self.s1Difference = None
        self.itemBasedResults = None
        self.userBasedResults = None
        self.s1BasedResults = None


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
        recommendation = np.zeros(self.width)
        recommendation[nonzero] = (ratingsSum[nonzero] + b * globalAvg) / (ratingsNum[nonzero] + b) #calculate movie avrage
        print("Time to calculate movie average: {0:.2f}s".format(time.time() - start))

        dtype = [('title', np.dtype((str, 100))), ('rating', float)]  # define numpy data type
        if user is not None: #check if user was given
            userIndex = np.where(self.userID == user)[0][0] #finde user index
            seenMovies = ~np.isnan(self.userRatings[userIndex, :]) #finde seen movies
            rec = np.array(list(zip(self.movieTitle[seenMovies], recommendation[seenMovies])), dtype=dtype) #print average of those seen movies
        else:
            rec = np.array(list(zip(self.movieTitle, recommendation)), dtype=dtype)  # combine movie rating with title

        rec = np.sort(rec, order='rating')[-n:][::-1] #sort, take last n, reverse
        for r in rec:
            print(r)

    def reduceDB(self, minUser, minItem):
        self.usableMovies = np.sum(~np.isnan(self.userRatings), axis=0) > minItem  # get movies with atleast m ratings
        tmp = self.userRatings[:, self.usableMovies]  # save them
        self.usableUsers = np.sum(~np.isnan(tmp), axis=1) > minUser  # get users with atleast m ratings
        self.X = np.copy(tmp[self.usableUsers, :])  # save them

        self.userAve = np.array([np.nanmean(self.X, axis=1)])  # calculate mean for every user
        self.userAve = self.userAve.T  # transpose for subtracting
        self.X = self.X - self.userAve  # subtract mean of user from their ratings

        self.newWidth = self.X.shape[1]  # save new width
        self.newHeight = self.X.shape[0]  # save new height
        self.newUser = self.userID[self.usableUsers]  # save reduced userID table
        self.newMovies = self.movieID[self.usableMovies]  # save reduced movieID table

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              ITEM BASED PREDICTION                                                                   #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def similarityFun(self, i1, i2, K):
        notNan = ~np.isnan(i1) & ~np.isnan(i2) #finde where both are numbers
        i1 = i1[notNan] #remember, for speed
        i2 = i2[notNan] #remember, for speed
        cov = np.dot(i1,i2)
        var1 = np.dot(i1,i1)
        var2 = np.dot(i2,i2)
        return cov/(sqrt(var1)*sqrt(var2)+K)

    def ItemBasedPredictionFit(self, K, threshold, users=0, items=300):
        start = time.time()

        self.reduceDB(users, items)

        self.itemSimilarity = np.zeros((self.newWidth,self.newWidth)) #crate ampty array
        for i1, i2 in combinations(range(self.newWidth), 2): #get all combinations of movies
            s = self.similarityFun(self.X[:, i1], self.X[:, i2], K) #calculate similarity
            if s < threshold: s = 0
            self.itemSimilarity[i1, i2] = s #save it
            self.itemSimilarity[i2, i1] = s

        print("Time to calculate similarity: {0:.2f}s".format(time.time() - start))

    def _ItemBasedRecommendUser(self, user, recSeen):
        start = time.time()
        userIndex = np.where(self.newUser == user)[0][0]  # finde user index
        notRatedMovies = np.argwhere(np.isnan(self.X[userIndex, :])).ravel()
        ratedMovies = ~np.isnan(self.X[userIndex, :])
        userRatingValue = self.X[userIndex, ratedMovies]

        moviesRecommend = np.zeros(self.newWidth)
        for i in notRatedMovies:
            similarity = self.itemSimilarity[i, ratedMovies]
            similaritySum = np.sum(similarity)
            if similaritySum == 0:
                rating = 0
            else:
                rating = np.dot(userRatingValue, similarity) / similaritySum
            moviesRecommend[i] = self.userAve[userIndex] + rating

        print("Time to finde similar movies: {0:.2f}s".format(time.time() - start))

        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.itemBasedResults = []
        for i in range(self.newWidth):
            if recSeen:
                if notSeenMovies[i]:
                    self.itemBasedResults.append((self.newMovies[i], moviesRecommend[i]))
                else:
                    self.itemBasedResults.append((self.newMovies[i], self.X[userIndex, i]))
            else:
                if notSeenMovies[i]:
                    self.itemBasedResults.append((self.newMovies[i], moviesRecommend[i]))

    def ItemBasedRecommendUser(self, user, n=20, recSeen=False):
        self._ItemBasedRecommendUser(user, recSeen)

        self.itemBasedResults.sort(key=lambda tup: -tup[1]) #sort them
        if len(self.itemBasedResults) > n:
            for r in self.itemBasedResults[:n]:
                print("{} - {:.2f}".format(self.movieTitle[self.movieDict[r[0]]], r[1]))
        else:
            for r in self.itemBasedResults:
                print("{} - {:.2f}".format(self.movieTitle[self.movieDict[r[0]]], r[1]))

    def ItemBasedBest(self, n=20):
        start = time.time()
        result = []
        for i1, i2 in combinations(range(self.newWidth), 2):
            result.append((i1, i2, self.itemSimilarity[i1, i2])) #save all similarities

        result = sorted(result, key=lambda tup: -tup[2])[:n] #sort them an get the best
        print("Time to finde similar movies: {0:.2f}s".format(time.time() - start))

        for r in result: #print the best
            i1 = self.newMovies[r[0]] #get index
            m1 = self.movieTitle[self.movieDict[i1]] #get title
            i2 = self.newMovies[r[1]] #get index
            m2 = self.movieTitle[self.movieDict[i2]] #get title
            print((m1, m2, r[2]))

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              USER BASED PREDICTION                                                                   #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def UserBasedPredictionFit(self, K, threshold, users=500, items=0):
        start = time.time()

        self.reduceDB(users, items)

        self.userSimilarity = np.zeros((self.newHeight,self.newHeight)) #crate ampty array
        for u1, u2 in combinations(range(self.newHeight), 2): #get all combinations of movies
            s = self.similarityFun(self.X[u1, :], self.X[u2, :], K) #calculate similarity
            if s < threshold: s = 0
            self.userSimilarity[u1, u2] = s #save it
            self.userSimilarity[u2, u1] = s

        print("Time to calculate similarity: {0:.2f}s".format(time.time() - start))

    def _UserBasedRecommendUser(self, user, recSeen):
        start = time.time()
        userIndex = np.where(self.newUser == user)[0][0]  # finde user index
        notRatedMovies = np.argwhere(np.isnan(self.X[userIndex, :])).ravel()

        moviesRecommend = np.zeros(self.newWidth)
        for i in notRatedMovies:
            ratedByUsers = ~np.isnan(self.X[:, i])
            similarity = self.userSimilarity[userIndex, ratedByUsers]
            similaritySum = np.sum(similarity)
            if similaritySum == 0:
                rating = 0
            else:
                rating = np.dot(self.X[ratedByUsers, i], similarity) / similaritySum
            moviesRecommend[i] = self.userAve[userIndex] + rating

        print("Time to finde similar movies: {0:.2f}s".format(time.time() - start))

        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.userBasedResults = []
        for i in range(self.newWidth):
            if recSeen:
                if notSeenMovies[i]:
                    self.userBasedResults.append((self.newMovies[i], moviesRecommend[i]))
                else:
                    self.userBasedResults.append((self.newMovies[i], self.X[userIndex, i]))
            else:
                if notSeenMovies[i]:
                    self.userBasedResults.append((self.newMovies[i], moviesRecommend[i]))

    def UserBasedRecommendUser(self, user, n=20, recSeen=False):
        if user not in self.newUser:
            print("User has not enough ratings!")
            return

        self._UserBasedRecommendUser(user, recSeen)

        self.userBasedResults.sort(key=lambda tup: -tup[1]) #sort them
        if len(self.userBasedResults) > n:
            for r in self.userBasedResults[:n]:
                print("{} - {:.2f}".format(self.movieTitle[self.movieDict[r[0]]], r[1]))
        else:
            for r in self.userBasedResults:
                print("{} - {:.2f}".format(self.movieTitle[self.movieDict[r[0]]], r[1]))

    def UserBasedBest(self, n):
        start = time.time()
        result = []
        for u1, u2 in combinations(range(self.newHeight), 2):
            result.append((u1, u2, self.userSimilarity[u1, u2])) #save all similarities

        result = sorted(result, key=lambda tup: -tup[2])[:n] #sort them an get the best
        print("Time to finde similar movies: {0:.2f}s".format(time.time() - start))

        for r in result: #print the best
            u1 = self.newUser[r[0]] #get id
            u2 = self.newUser[r[1]] #get id
            print((u1, u2, r[2]))

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              SLOPE ONE PREDICTION                                                                    #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def differenceFunS1(self, i1, i2):
        notNan = ~np.isnan(i1) & ~np.isnan(i2) #finde where both are numbers
        itemSum = np.sum(i2[notNan] - i1[notNan])
        numCommon = np.sum(notNan)
        if numCommon == 0:
            return 0, 0
        else:
            return itemSum/numCommon, numCommon

    def SlopeOnePredictionFit(self, users=0, items=400):
        start = time.time()

        self.reduceDB(users, items)

        self.s1DifferenceNum = np.zeros((self.newWidth,self.newWidth))
        self.s1Difference = np.zeros((self.newWidth,self.newWidth)) #crate ampty array
        for i1, i2 in combinations(range(self.newWidth), 2): #get all combinations of movies
            d, n = self.differenceFunS1(self.X[i1, :], self.X[i2, :]) #calculate similarity
            self.s1Difference[i1, i2] = d #save it
            self.s1DifferenceNum[i1, i2] = n  # save it
            self.s1Difference[i2, i1] = -d
            self.s1DifferenceNum[i2, i1] = n  # save it

        print("Time to calculate similarity: {0:.2f}s".format(time.time() - start))

    def _SlopeOneRecommendUser(self, user, recSeen):
        start = time.time()
        userIndex = np.where(self.newUser == user)[0][0]  # finde user index
        notRatedMovies = np.argwhere(np.isnan(self.X[userIndex, :])).ravel()
        ratedMovies = ~np.isnan(self.X[userIndex, :])
        userRatingValue = self.X[userIndex, ratedMovies]

        moviesRecommend = np.zeros(self.newWidth)
        for i in notRatedMovies:
            difference = self.s1Difference[i, ratedMovies]
            differenceNum = self.s1DifferenceNum[i, ratedMovies]
            numSum = np.sum(differenceNum)
            if numSum == 0:
                rating = 0
            else:
                rating = np.sum((userRatingValue-difference)*differenceNum) / numSum
            moviesRecommend[i] = self.userAve[userIndex] + rating

        print("Time to finde similar movies: {0:.2f}s".format(time.time() - start))

        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.s1BasedResults = []
        for i in range(self.newWidth):
            if recSeen:
                if notSeenMovies[i]:
                    self.s1BasedResults.append((self.newMovies[i], moviesRecommend[i]))
                else:
                    self.s1BasedResults.append((self.newMovies[i], self.X[userIndex, i]))
            else:
                if notSeenMovies[i]:
                    self.s1BasedResults.append((self.newMovies[i], moviesRecommend[i]))

    def SlopeOneRecommendUser(self, user, n=20, recSeen=False):
        if user not in self.newUser:
            print("User has not enough ratings!")
            return

        self._SlopeOneRecommendUser(user, recSeen)

        self.s1BasedResults.sort(key=lambda tup: -tup[1]) #sort them
        if len(self.s1BasedResults) > n:
            for r in self.s1BasedResults[:n]:
                print("{} - {:.2f}".format(self.movieTitle[self.movieDict[r[0]]], r[1]))
        else:
            for r in self.s1BasedResults:
                print("{} - {:.2f}".format(self.movieTitle[self.movieDict[r[0]]], r[1]))