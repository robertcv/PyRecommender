import numpy as np
from math import sqrt
from itertools import combinations, product
import time
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


class Recommender(object):
    def __init__(self):
        # for db
        self.movieID = None
        self.movieTitle = []
        self.movieDict = {}
        self.userID = None
        self.userRatings = None
        # for reduced db
        self.userAve = None
        self.X = None
        self.Width = None
        self.Height = None
        self.newUser = None
        self.newMovies = None
        # similarity results
        self.itemSimilarity = None
        self.userSimilarity = None
        self.s1DifferenceNum = None
        self.s1Difference = None
        # ratings for all movies from user
        self.itemAllUserRatings = None
        self.userAllUserRatings = None
        self.s1AllUserRatings = None
        self.hybridAllUserRatings = None
        self.nmfAllUserRatings = None
        # best rated movies for user
        self.itemBasedResults = None
        self.userBasedResults = None
        self.s1BasedResults = None
        self.hybridBasedResults = None
        self.nmfBasedResults = None
        # for nmf
        self.W = None
        self.H = None
        # for nb
        self.nbMovieID = None
        self.nbTagID = None
        self.nbTagName = None
        self.nbTagDict = {}
        self.nbMovieTags = None
        self.nbBasedResults = None

    def parseMovieDB(self, movie_file, ratedmovies_file):
        start = time.time()
        movie = open(movie_file, 'rt', encoding='utf-8')
        movieID = []
        movieTitle = []

        first_line = movie.readline()  # header line
        i = 0
        for line in movie:
            ln = line.strip().split('\t')
            movieID.append(ln[0])
            movieTitle.append(ln[1])  # save title
            self.movieDict[ln[0]] = i  # save position of movie
            i += 1

        width = len(movieID)  # set the width of the rating matix

        # code for finding the number of users
        """
        ratedmovies = open(ratedmovies_file, 'rt', encoding='utf-8')
        userSet = set()
        first_line = ratedmovies.readline()
        for line in ratedmovies:
            ln = line.strip().split('\t')
            userSet.add(ln[0])
        """
        height = 2114  # 2113 #len(userSet)  set the height of the rating matix
        # for speed its hardcoded

        ratedmovies = open(ratedmovies_file, 'rt', encoding='utf-8')
        userID = []  # save user ids
        self.userRatings = np.full((height, width), np.nan)  # crate an empty matrix

        first_line = ratedmovies.readline()  # header line

        ln = ratedmovies.readline().split('\t')  # reade first line
        curentU = ln[0]  # remember curent user
        self.userRatings[0, self.movieDict[ln[1]]] = ln[2]  # set first rating

        j = 0
        for line in ratedmovies:
            ln = line.strip().split('\t')
            if ln[0] == curentU:  # if the user is the same just save te rating for the movie
                self.userRatings[j, self.movieDict[ln[1]]] = ln[2]
            else:  # if its a new user
                userID.append(curentU)  # save the old
                curentU = ln[0]  # save new user
                j += 1
                self.userRatings[j, self.movieDict[ln[1]]] = ln[2]  # save rating

        userID.append(curentU)  # save the last user

        self.userID = np.array(userID)
        self.movieID = np.array(movieID)
        self.movieTitle = np.array(movieTitle)
        print("Time to load data: {0:.2f}s".format(time.time() - start))

    def reduceDB(self, minUser, minItem):
        usableMovies = np.sum(~np.isnan(self.userRatings), axis=0) > minItem  # get movies with atleast m ratings
        tmp = self.userRatings[:, usableMovies]  # save them
        usableUsers = np.sum(~np.isnan(tmp), axis=1) > minUser  # get users with atleast m ratings
        self.X = np.copy(tmp[usableUsers, :])  # save them

        self.userAve = np.array([np.nanmean(self.X, axis=1)])  # calculate mean for every user
        self.userAve = self.userAve.T  # transpose for subtracting
        self.X = self.X - self.userAve  # subtract mean of user from their ratings

        self.Width = self.X.shape[1]  # save new width
        self.Height = self.X.shape[0]  # save new height
        self.newUser = self.userID[usableUsers]  # save reduced userID table
        self.newMovies = self.movieID[usableMovies]  # save reduced movieID table

    def printRecommendation(self, results, n):
        results.sort(key=lambda tup: -tup[1])  # sort them
        for r in results[:n]:
            print("{} - {:.2f}".format(self.movieTitle[self.movieDict[r[0]]], r[1])) # print results


########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              AVERAGE PREDICTION                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def averagePrediction(self, b, n=20, user=None, users=0, items=0, prepareDB=True):
        start = time.time()

        if prepareDB:
            self.reduceDB(users, items)

        ratingsSum = np.nansum(self.userRatings, axis=0)  # add all ratings of movies
        ratingsNum = np.sum(~np.isnan(self.userRatings), axis=0)  # count the number of given ratings
        globalAvg = np.sum(ratingsSum) / np.sum(ratingsNum)  # calculate global avrage

        nonzero = ratingsNum != 0  # finde all movies wich have atlest one rating
        averageResults = np.zeros(self.Width)
        averageResults[nonzero] = (ratingsSum[nonzero] + b * globalAvg) / (
            ratingsNum[nonzero] + b)  # calculate movie avrage
        print("Time to calculate movie average: {0:.2f}s".format(time.time() - start))

        if user is not None:  # check if user was given
            userIndex = np.where(self.userID == user)[0][0]  # finde user index
            seenMovies = ~np.isnan(self.userRatings[userIndex, :])  # finde seen movies
            self.printRecommendation(list(zip(self.movieID[seenMovies], averageResults[seenMovies])), n)
        else:
            self.printRecommendation(list(zip(self.movieID, averageResults)), n)


########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              ITEM BASED PREDICTION                                                                   #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def itemSimilarityFun(self, i1, i2, K, threshold):
        item1, item2 = self.X[:, i1], self.X[:, i2]
        notNan = ~np.isnan(item1) & ~np.isnan(item2)  # finde where both are numbers
        item1 = item1[notNan]  # remember, for speed
        item2 = item2[notNan]  # remember, for speed
        cov = np.dot(item1, item2)
        var1 = np.dot(item1, item1)
        var2 = np.dot(item2, item2)
        s = cov / (sqrt(var1) * sqrt(var2) + K)
        if s < threshold: s = 0
        self.itemSimilarity[i1, i2] = s  # save it
        self.itemSimilarity[i2, i1] = s


    def ItemBasedPredictionFit(self, K, threshold, users=0, items=300, prepareDB=True):
        start = time.time()

        if prepareDB:
            self.reduceDB(users, items)

        self.itemSimilarity = np.zeros((self.Width, self.Width))  # crate empty array
        for i1, i2 in combinations(range(self.Width), 2):  # get all combinations of movies
            self.itemSimilarityFun(i1, i2, K, threshold)

        print("Time to calculate item similarity: {0:.2f}s".format(time.time() - start))

    def _ItemBasedRecommendUser(self, user, recSeen):
        userIndex = np.where(self.newUser == user)[0][0]  # find user index
        notRatedMovies = np.argwhere(np.isnan(self.X[userIndex, :])).ravel()
        ratedMovies = ~np.isnan(self.X[userIndex, :])
        userRatingValue = self.X[userIndex, ratedMovies]

        self.itemAllUserRatings = np.zeros(self.Width)
        for i in notRatedMovies:
            similarity = self.itemSimilarity[i, ratedMovies]  # get similarity
            similaritySum = np.sum(similarity)
            if similaritySum == 0:  # for zero division
                rating = 0
            else:
                rating = np.dot(userRatingValue, similarity) / similaritySum # calculate rating
            self.itemAllUserRatings[i] = self.userAve[userIndex] + rating

        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.itemBasedResults = []
        for i in range(self.Width):
            if recSeen:
                if notSeenMovies[i]:
                    self.itemBasedResults.append((self.newMovies[i], self.itemAllUserRatings[i]))
                else:
                    self.itemBasedResults.append((self.newMovies[i], self.X[userIndex, i]))
            else:
                if notSeenMovies[i]:
                    self.itemBasedResults.append((self.newMovies[i], self.itemAllUserRatings[i]))

    def ItemBasedRecommendUser(self, user, n=20, recSeen=False):
        if user not in self.newUser:
            print("User has not enough ratings!")
            return
        self._ItemBasedRecommendUser(user, recSeen)
        self.printRecommendation(self.itemBasedResults, n)

    def ItemBasedBest(self, n=20):
        result = []
        for i1, i2 in combinations(range(self.Width), 2):
            result.append((i1, i2, self.itemSimilarity[i1, i2]))  # save all similarities

        result = sorted(result, key=lambda tup: -tup[2])[:n]  # sort them an get the best

        for r in result:  # print the best
            i1 = self.newMovies[r[0]]  # get index
            m1 = self.movieTitle[self.movieDict[i1]]  # get title
            i2 = self.newMovies[r[1]]  # get index
            m2 = self.movieTitle[self.movieDict[i2]]  # get title
            print((m1, m2, r[2]))

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              USER BASED PREDICTION                                                                   #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################
    def userSimilarityFun(self, u1, u2, K, threshold):
        user1, user2 = self.X[u1, :], self.X[u2, :]
        notNan = ~np.isnan(user1) & ~np.isnan(user2)  # finde where both are numbers
        user1 = user1[notNan]  # remember, for speed
        user2 = user2[notNan]  # remember, for speed
        cov = np.dot(user1, user2)
        var1 = np.dot(user1, user1)
        var2 = np.dot(user2, user2)
        s = cov / (sqrt(var1) * sqrt(var2) + K)
        if s < threshold: s = 0
        self.userSimilarity[u1, u2] = s  # save it
        self.userSimilarity[u2, u1] = s

    def UserBasedPredictionFit(self, K, threshold, users=500, items=0, prepareDB=True):
        start = time.time()

        if prepareDB:
            self.reduceDB(users, items)

        self.userSimilarity = np.zeros((self.Height, self.Height))  # crate ampty array
        for u1, u2 in combinations(range(self.Height), 2):  # get all combinations of movies
            self.userSimilarityFun(u1, u2, K, threshold)  # calculate similarity

        print("Time to calculate user similarity: {0:.2f}s".format(time.time() - start))

    def _UserBasedRecommendUser(self, user, recSeen):
        userIndex = np.where(self.newUser == user)[0][0]  # finde user index
        notRatedMovies = np.argwhere(np.isnan(self.X[userIndex, :])).ravel()

        self.userAllUserRatings = np.zeros(self.Width)
        for i in notRatedMovies:
            ratedByUsers = ~np.isnan(self.X[:, i])
            similarity = self.userSimilarity[userIndex, ratedByUsers]
            similaritySum = np.sum(similarity)
            if similaritySum == 0: # for zero division
                rating = 0
            else:
                rating = np.dot(self.X[ratedByUsers, i], similarity) / similaritySum  # calculate rating
            self.userAllUserRatings[i] = self.userAve[userIndex] + rating

        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.userBasedResults = []
        for i in range(self.Width):
            if recSeen:
                if notSeenMovies[i]:
                    self.userBasedResults.append((self.newMovies[i], self.userAllUserRatings[i]))
                else:
                    self.userBasedResults.append((self.newMovies[i], self.X[userIndex, i]))
            else:
                if notSeenMovies[i]:
                    self.userBasedResults.append((self.newMovies[i], self.userAllUserRatings[i]))

    def UserBasedRecommendUser(self, user, n=20, recSeen=False):
        if user not in self.newUser:
            print("User has not enough ratings!")
            return

        self._UserBasedRecommendUser(user, recSeen)
        self.printRecommendation(self.userBasedResults, n)

    def UserBasedBest(self, n):
        result = []
        for u1, u2 in combinations(range(self.Height), 2):
            result.append((u1, u2, self.userSimilarity[u1, u2]))  # save all similarities

        result = sorted(result, key=lambda tup: -tup[2])[:n]  # sort them an get the best

        for r in result:  # print the best
            u1 = self.newUser[r[0]]  # get id
            u2 = self.newUser[r[1]]  # get id
            print((u1, u2, r[2]))

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              SLOPE ONE PREDICTION                                                                    #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def differenceFunS1(self, i1, i2):
        item1, item2 = self.X[:, i1], self.X[:, i2]
        notNan = ~np.isnan(item1) & ~np.isnan(item2)  # finde where both are numbers
        itemSum = np.sum(item2[notNan] - item1[notNan])
        numCommon = np.sum(notNan)
        if numCommon == 0:
            d, n = 0, 0
        else:
            d, n = itemSum / numCommon, numCommon
        self.s1Difference[i1, i2] = d  # save it
        self.s1DifferenceNum[i1, i2] = n  # save it
        self.s1Difference[i2, i1] = -d
        self.s1DifferenceNum[i2, i1] = n  # save it


    def SlopeOnePredictionFit(self, users=0, items=400, prepareDB=True):
        start = time.time()

        if prepareDB:
            self.reduceDB(users, items)

        self.s1DifferenceNum = np.zeros((self.Width, self.Width))
        self.s1Difference = np.zeros((self.Width, self.Width))  # crate empty array
        for i1, i2 in combinations(range(self.Width), 2):  # get all combinations of movies
            self.differenceFunS1(i1, i2)  # calculate similarity

        print("Time to calculate slope on: {0:.2f}s".format(time.time() - start))

    def _SlopeOneRecommendUser(self, user, recSeen):
        userIndex = np.where(self.newUser == user)[0][0]  # finde user index
        notRatedMovies = np.argwhere(np.isnan(self.X[userIndex, :])).ravel()
        ratedMovies = ~np.isnan(self.X[userIndex, :])
        userRatingValue = self.X[userIndex, ratedMovies]

        self.s1AllUserRatings = np.zeros(self.Width)
        for i in notRatedMovies:
            difference = self.s1Difference[i, ratedMovies]
            differenceNum = self.s1DifferenceNum[i, ratedMovies]
            numSum = np.sum(differenceNum)
            if numSum == 0:
                rating = 0
            else:
                rating = np.sum((userRatingValue - difference) * differenceNum) / numSum
            self.s1AllUserRatings[i] = self.userAve[userIndex] + rating

        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.s1BasedResults = []
        for i in range(self.Width):
            if recSeen:
                if notSeenMovies[i]:
                    self.s1BasedResults.append((self.newMovies[i], self.s1AllUserRatings[i]))
                else:
                    self.s1BasedResults.append((self.newMovies[i], self.X[userIndex, i]))
            else:
                if notSeenMovies[i]:
                    self.s1BasedResults.append((self.newMovies[i], self.s1AllUserRatings[i]))

    def SlopeOneRecommendUser(self, user, n=20, recSeen=False):
        if user not in self.newUser:
            print("User has not enough ratings!")
            return
        self._SlopeOneRecommendUser(user, recSeen)
        self.printRecommendation(self.s1BasedResults, n)

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              EVALUET                                                                                 #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def testPredictor(self, name, fitFnc, recommenderFun, userRatings):
        RMSE = 0
        MAE = 0
        eNum = 0
        precision = 0
        recall = 0
        tpNum = 0

        fitFnc(self.K, self.threshold, prepareDB=False) #fit data
        userNum = 0
        for u in np.argwhere(self.testUsers).ravel():
            recommenderFun(self.newUser[u], recSeen=False) #recommend for test users
            testItemsRatedUser = self.testItemsRated[userNum, :] # get recommendet ratings
            calculatedRatings = userRatings(testItemsRatedUser)
            realRatings = self.testItems[userNum, testItemsRatedUser] #get real ratings
            ratingDifference = (calculatedRatings - self.userAve[u]) - realRatings
            relativeCalculatedR = calculatedRatings - self.userAve[u]
            userNum += 1

            #calculate RMSE and MAE
            if np.sum(testItemsRatedUser) != 0:
                RMSE += sqrt(np.sum(ratingDifference * ratingDifference) / np.sum(testItemsRatedUser))
                MAE += np.sum(np.abs(ratingDifference)) / np.sum(testItemsRatedUser)
                eNum += 1

            #calculate precision and recall
            tAboveAvg = realRatings > 0
            tBelowAvg = realRatings <= 0

            rAboveAvg = relativeCalculatedR > 0
            rBelowAvg = relativeCalculatedR <= 0

            tp = np.sum(tAboveAvg & rAboveAvg)
            fp = np.sum(tBelowAvg & rAboveAvg)
            fn = np.sum(tAboveAvg & rBelowAvg)

            if tp != 0:
                precision += tp / (tp + fp)
                recall += tp / (tp + fn)
                tpNum += 1

        precision = precision / tpNum
        recall = recall / tpNum

        print(name + " based:")
        print("\tRMSE: {:.2f}".format(RMSE / eNum))
        print("\tMAE: {:.2f}".format(MAE / eNum))
        print("\tprecision: {:.2f}".format(precision))
        print("\trecall: {:.2f}".format(recall))
        print("\tF1: {:.2f}".format(2 * (precision * recall) / (recall + precision)))

    def Evaluet(self, K, threshold):
        self.K = K
        self.threshold = threshold
        self.reduceDB(200, 300)
        testUserNum = int(self.Height * 0.3) # 30% users for testing
        self.testUsers = np.concatenate((np.zeros(self.Height - testUserNum, dtype=bool), np.ones(testUserNum, dtype=bool)))
        np.random.shuffle(self.testUsers) # get random test users
        testItemNum = int(self.Width * 0.5) # 50% items for calculating acurasety
        self.testItems = np.copy(self.X[self.testUsers, :])
        self.testItemsRated = ~np.isnan(self.X[self.testUsers, :])
        self.X[self.testUsers, :testItemNum] = np.nan
        self.testItemsRated[:, testItemNum:] = False

        self.testPredictor("Item", self.ItemBasedPredictionFit, self._ItemBasedRecommendUser,
                           lambda x: self.itemAllUserRatings[x])

        self.testPredictor("User", self.UserBasedPredictionFit, self._UserBasedRecommendUser,
                           lambda x: self.userAllUserRatings[x])

        self.testPredictor("Slope one", self.SlopeOnePredictionFit, self._SlopeOneRecommendUser,
                           lambda x: self.s1AllUserRatings[x])

        #self.testPredictor("Hybrid", self.HybridPredictionFit, self._HybridRecommendUser,
        #                   lambda x: self.hybridAllUserRatings[x])

        #self.testPredictor("NMF", self.MatrixFactorizationFit, self._MatrixFactorizationRecommendUser,
        #                  lambda x: self.nmfAllUserRatings[x]) #add k, threshold to fit function

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              HYBRID PREDICTOR                                                                        #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def HybridPredictionFit(self, K, threshold, users=300, items=400, prepareDB=True):
        start = time.time()
        if prepareDB:
            self.reduceDB(users, items)

        # run fit functions
        self.ItemBasedPredictionFit(K, threshold, prepareDB=False)
        self.UserBasedPredictionFit(K, threshold, prepareDB=False)
        self.SlopeOnePredictionFit(prepareDB=False)
        print("Time to calculate hybrid prediction: {0:.2f}s".format(time.time() - start))

    def _HybridRecommendUser(self, user, recSeen):
        self._ItemBasedRecommendUser(user, recSeen)
        self._UserBasedRecommendUser(user, recSeen)
        self._SlopeOneRecommendUser(user, recSeen)
        #calculate hybrid result
        self.hybridAllUserRatings = 1 / 2 * self.itemAllUserRatings + 2 / 6 * self.userAllUserRatings + 1 / 6 * self.s1AllUserRatings
        userIndex = np.where(self.newUser == user)[0][0]  # finde user index
        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.hybridBasedResults = []
        for i in range(self.Width):
            if recSeen:
                if notSeenMovies[i]:
                    self.hybridBasedResults.append((self.newMovies[i], self.hybridAllUserRatings[i]))
                else:
                    self.hybridBasedResults.append((self.newMovies[i], self.X[userIndex, i]))
            else:
                if notSeenMovies[i]:
                    self.hybridBasedResults.append((self.newMovies[i], self.hybridAllUserRatings[i]))

    def HybridRecommendUser(self, user, n=20, recSeen=False):
        if user not in self.newUser:
            print("User has not enough ratings!")
            return

        self._HybridRecommendUser(user, recSeen)
        self.printRecommendation(self.hybridBasedResults, n)

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              MATRIX FACTORIZATION                                                                    #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def MatrixFactorizationFit(self, rank=5, max_iter=20, eta=0.001, prepareDB=True, users=300, items=500):
        start = time.time()
        if prepareDB:
            self.reduceDB(users, items)

        W = np.random.rand(self.Height, rank)
        H = np.random.rand(self.Width, rank)

        # Indices to model variables
        w_vars = list(product(range(self.Height), range(rank)))
        h_vars = list(product(range(self.Width), range(rank)))

        # Indices to nonzero rows/columns
        nzcols = dict([(j, np.nonzero(~np.isnan(self.X[:, j]))[0]) for j in range(self.Width)])
        nzrows = dict([(i, np.nonzero(~np.isnan(self.X[i, :]))[0]) for i in range(self.Height)])

        # Errors
        error = np.zeros((max_iter,))

        for t in range(max_iter):
            np.random.shuffle(w_vars)
            np.random.shuffle(h_vars)

            for i, k in w_vars:
                # Calculate gradient and update W[i, k]
                W[i, k] = max(W[i, k] + eta*np.sum([self.X[i, j] - W[i, :].dot(H[j, :]) for j in nzrows[i]])*W[i, k],0)

            for j, k in h_vars:
                # Calculate gradient and update H[j, k]
                H[j, k] = max(H[j, k] + eta*np.sum([self.X[i, j] - W[i, :].dot(H[j, :]) for i in nzcols[j]])*H[j, k],0)

            #error[t] = np.linalg.norm((self.X - W.dot(H.T))[~np.isnan(self.X)]) ** 2
            #print(t, error[t])

        self.W = W
        self.H = H

        print("Time to calculate factorization matrix: {0:.2f}s".format(time.time() - start))


    def _MatrixFactorizationRecommendUser(self, user, recSeen):
        userIndex = np.where(self.newUser == user)[0][0]  # finde user index

        self.nmfAllUserRatings = self.userAve[userIndex] + self.W.dot(self.H.T)[userIndex, :]

        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.nmfBasedResults = []
        for i in range(self.Width):
            if recSeen:
                if notSeenMovies[i]:
                    self.nmfBasedResults.append((self.newMovies[i], self.nmfAllUserRatings[i]))
                else:
                    self.nmfBasedResults.append((self.newMovies[i], self.X[userIndex, i]))
            else:
                if notSeenMovies[i]:
                    self.nmfBasedResults.append((self.newMovies[i], self.nmfAllUserRatings[i]))

    def MatrixFactorizationRecommendUser(self, user, n=20, recSeen=False):
        if user not in self.newUser:
            print("User has not enough ratings!")
            return

        self._MatrixFactorizationRecommendUser(user, recSeen)
        self.printRecommendation(self.nmfBasedResults, n)

    def MatrikFactorizationGraph(self, n=20):
        maxIndex = np.sum(~np.isnan(self.X), axis=0).argsort()[-n:][::-1]
        row1 = self.H.T[0, maxIndex]
        row2 = self.H.T[1, maxIndex]
        movies = self.newMovies[maxIndex]
        movieTitles = [self.movieTitle[self.movieDict[m]] for m in movies]

        for i in range(len(movies)):
            plt.annotate(movieTitles[i], xy=(row1[i], row2[i]), xytext=(row1[i], row2[i]),
                         horizontalalignment='center',
                         verticalalignment='center', )
        plt.ylim([-0.5, 2.5])
        plt.xlim([-0.5, 2.5])
        plt.show()





########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              NAIVE BAYES PREDICTOR                                                                   #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def parseTageDB(self, movie_tags_file, tags_file):
        start = time.time()
        #get tag info
        tags = open(tags_file, 'rt', encoding='utf-8')
        tagID = []
        tagName = []

        first_line = tags.readline()  # header line
        i = 0
        for line in tags:
            ln = line.strip().split('\t')
            tagID.append(ln[0])
            tagName.append(ln[1])
            self.nbTagDict[ln[0]] = i
            i += 1

        width = len(tagID)
        #get movies with tags
        movie_tags = open(movie_tags_file, 'rt', encoding='utf-8')
        movieSet = set()
        first_line = movie_tags.readline()
        for line in movie_tags:
            ln = line.strip().split('\t')
            movieSet.add(ln[0])

        height = len(movieSet)  #set the height of the rating matix

        movie_tags = open(movie_tags_file, 'rt', encoding='utf-8')
        movieID = []
        self.nbMovieTags = np.zeros((height, width))  # crate an empty matrix

        first_line = movie_tags.readline()

        ln = movie_tags.readline().split('\t')
        curentM = ln[0]
        self.nbMovieTags[0, self.nbTagDict[ln[1]]] = 1

        j = 0
        for line in movie_tags:
            ln = line.strip().split('\t')
            if ln[0] == curentM:
                self.nbMovieTags[j, self.nbTagDict[ln[1]]] = 1
            else:
                movieID.append(curentM)
                curentM = ln[0]
                j += 1
                self.nbMovieTags[j, self.nbTagDict[ln[1]]] = 1

        movieID.append(curentM)

        self.nbTagID = np.array(tagID)
        self.nbTagName = np.array(tagName)
        self.nbMovieID = np.array(movieID)
        print("Time to load data: {0:.2f}s".format(time.time() - start))

    def NaiveBayes(self, user, aboveAvg, n=10, tags=10, movies=10):
        self.reduceDB(0, 0)
        if user not in self.newUser:
            print("User has not enough ratings!")
            return
        # reduce tags matrix
        usableTags = np.sum(self.nbMovieTags, axis=0) > tags
        usableMovies = np.sum(self.nbMovieTags, axis=1) > movies
        self.nbMovieTags = self.nbMovieTags[usableMovies, :]
        self.nbMovieTags = self.nbMovieTags[:, usableTags]
        self.nbTagID = self.nbTagID[usableTags]
        self.nbMovieID = self.nbMovieID[usableMovies]

        userIndex = np.where(self.newUser == user)[0][0]
        userRatings = self.X[userIndex, :]
        userRated = ~np.isnan(userRatings)
        userRatings = np.nan_to_num(userRatings)
        aboveAvgRatings = userRatings > aboveAvg
        x = []
        y = []
        predict = []
        predictMovies = []
        for i in range(len(userRatings)):
            if self.newMovies[i] in self.nbMovieID:
                movieIndex = np.where(self.nbMovieID == self.newMovies[i])[0][0]
                if userRated[i]:
                    x.append(self.nbMovieTags[movieIndex, :])
                    if aboveAvgRatings[i]:
                        y.append(1)
                    else:
                        y.append(-1)
                else:
                    predict.append(self.nbMovieTags[movieIndex, :])
                    predictMovies.append(self.newMovies[i])

        x = np.array(x)
        y = np.array(y)
        predict = np.array(predict)

        clf = GaussianNB()
        clf.fit(x, y)
        prediction = clf.predict(predict)
        proba = clf.predict_proba(predict)
        self.nbBasedResults = []
        for i in range(len(prediction)):
            if prediction[i] == 1:
                self.nbBasedResults.append((predictMovies[i], proba[i, 1]))

        self.printRecommendation(self.nbBasedResults, n)