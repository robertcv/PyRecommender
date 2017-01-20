import numpy as np
from math import sqrt
from itertools import combinations
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

    def printRecommendation(self, results, n):
        results.sort(key=lambda tup: -tup[1])  # sort them
        if len(results) > n:
            for r in results[:n]:
                print("{} - {:.2f}".format(self.movieTitle[self.movieDict[r[0]]], r[1]))
        else:
            for r in results:
                print("{} - {:.2f}".format(self.movieTitle[self.movieDict[r[0]]], r[1]))


########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              AVERAGE PREDICTION                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    def averagePrediction(self, b, n=20, user=None):
        start = time.time()
        # recommends movies with best avrage ratings
        ratingsSum = np.nansum(self.userRatings, axis=0)  # add all ratings of movies
        ratingsNum = np.sum(~np.isnan(self.userRatings), axis=0)  # count the number of given ratings
        globalAvg = np.sum(ratingsSum) / np.sum(ratingsNum)  # calculate global avrage

        nonzero = ratingsNum != 0  # finde all movies wich have atlest one rating
        averageResults = np.zeros(self.width)
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

    def similarityFun(self, i1, i2, K):
        notNan = ~np.isnan(i1) & ~np.isnan(i2) #finde where both are numbers
        i1 = i1[notNan] #remember, for speed
        i2 = i2[notNan] #remember, for speed
        cov = np.dot(i1,i2)
        var1 = np.dot(i1,i1)
        var2 = np.dot(i2,i2)
        return cov/(sqrt(var1)*sqrt(var2)+K)

    def ItemBasedPredictionFit(self, K, threshold, users=0, items=300, prepareDB=True):
        start = time.time()

        if prepareDB:
            self.reduceDB(users, items)

        self.itemSimilarity = np.zeros((self.newWidth,self.newWidth)) #crate ampty array
        for i1, i2 in combinations(range(self.newWidth), 2): #get all combinations of movies
            s = self.similarityFun(self.X[:, i1], self.X[:, i2], K) #calculate similarity
            if s < threshold: s = 0
            self.itemSimilarity[i1, i2] = s #save it
            self.itemSimilarity[i2, i1] = s

        print("Time to calculate similarity: {0:.2f}s".format(time.time() - start))

    def _ItemBasedRecommendUser(self, user, recSeen):
        userIndex = np.where(self.newUser == user)[0][0]  # finde user index
        notRatedMovies = np.argwhere(np.isnan(self.X[userIndex, :])).ravel()
        ratedMovies = ~np.isnan(self.X[userIndex, :])
        userRatingValue = self.X[userIndex, ratedMovies]

        self.itemAllUserRatings = np.zeros(self.newWidth)
        for i in notRatedMovies:
            similarity = self.itemSimilarity[i, ratedMovies]
            similaritySum = np.sum(similarity)
            if similaritySum == 0:
                rating = 0
            else:
                rating = np.dot(userRatingValue, similarity) / similaritySum
            self.itemAllUserRatings[i] = self.userAve[userIndex] + rating

        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.itemBasedResults = []
        for i in range(self.newWidth):
            if recSeen:
                if notSeenMovies[i]:
                    self.itemBasedResults.append((self.newMovies[i], self.itemAllUserRatings[i]))
                else:
                    self.itemBasedResults.append((self.newMovies[i], self.X[userIndex, i]))
            else:
                if notSeenMovies[i]:
                    self.itemBasedResults.append((self.newMovies[i], self.itemAllUserRatings[i]))

    def ItemBasedRecommendUser(self, user, n=20, recSeen=False):
        self._ItemBasedRecommendUser(user, recSeen)
        self.printRecommendation(self.itemBasedResults, n)

    def ItemBasedBest(self, n=20):
        result = []
        for i1, i2 in combinations(range(self.newWidth), 2):
            result.append((i1, i2, self.itemSimilarity[i1, i2])) #save all similarities

        result = sorted(result, key=lambda tup: -tup[2])[:n] #sort them an get the best

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

    def UserBasedPredictionFit(self, K, threshold, users=500, items=0, prepareDB=True):
        start = time.time()

        if prepareDB:
            self.reduceDB(users, items)

        self.userSimilarity = np.zeros((self.newHeight,self.newHeight)) #crate ampty array
        for u1, u2 in combinations(range(self.newHeight), 2): #get all combinations of movies
            s = self.similarityFun(self.X[u1, :], self.X[u2, :], K) #calculate similarity
            if s < threshold: s = 0
            self.userSimilarity[u1, u2] = s #save it
            self.userSimilarity[u2, u1] = s

        print("Time to calculate similarity: {0:.2f}s".format(time.time() - start))

    def _UserBasedRecommendUser(self, user, recSeen):
        userIndex = np.where(self.newUser == user)[0][0]  # finde user index
        notRatedMovies = np.argwhere(np.isnan(self.X[userIndex, :])).ravel()

        self.userAllUserRatings = np.zeros(self.newWidth)
        for i in notRatedMovies:
            ratedByUsers = ~np.isnan(self.X[:, i])
            similarity = self.userSimilarity[userIndex, ratedByUsers]
            similaritySum = np.sum(similarity)
            if similaritySum == 0:
                rating = 0
            else:
                rating = np.dot(self.X[ratedByUsers, i], similarity) / similaritySum
            self.userAllUserRatings[i] = self.userAve[userIndex] + rating

        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.userBasedResults = []
        for i in range(self.newWidth):
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
        for u1, u2 in combinations(range(self.newHeight), 2):
            result.append((u1, u2, self.userSimilarity[u1, u2])) #save all similarities

        result = sorted(result, key=lambda tup: -tup[2])[:n] #sort them an get the best

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

    def SlopeOnePredictionFit(self, users=0, items=400, prepareDB=True):
        start = time.time()

        if prepareDB:
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
        userIndex = np.where(self.newUser == user)[0][0]  # finde user index
        notRatedMovies = np.argwhere(np.isnan(self.X[userIndex, :])).ravel()
        ratedMovies = ~np.isnan(self.X[userIndex, :])
        userRatingValue = self.X[userIndex, ratedMovies]

        self.s1AllUserRatings = np.zeros(self.newWidth)
        for i in notRatedMovies:
            difference = self.s1Difference[i, ratedMovies]
            differenceNum = self.s1DifferenceNum[i, ratedMovies]
            numSum = np.sum(differenceNum)
            if numSum == 0:
                rating = 0
            else:
                rating = np.sum((userRatingValue-difference)*differenceNum) / numSum
            self.s1AllUserRatings[i] = self.userAve[userIndex] + rating


        notSeenMovies = np.isnan(self.X[userIndex, :])
        self.s1BasedResults = []
        for i in range(self.newWidth):
            if recSeen:
                if notSeenMovies[i]:
                    self.s1BasedResults.append((self.newMovies[i], self.s1AllUserRatings[i]))
                else:
                    self.s1BasedResults.append((self.newMovies[i], self.X[userIndex, i]))
            else:
                if notSeenMovies[i]:
                    self.s1BasedResults.append((self.newMovies[i], self.s1AllUserRatings[i]))

    def SlopeOneRecommendUser(self, user, n=20, recSeen=False):
        self._SlopeOneRecommendUser(user, recSeen)
        self.printRecommendation(self.s1BasedResults, n)


########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                              EVALUET                                                                                 #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################




    def Evaluet(self, K, threshold):
        self.reduceDB(100, 400)
        testUserNum = int(self.newHeight*0.3)
        testUsers = np.concatenate((np.zeros(self.newHeight-testUserNum, dtype=bool), np.ones(testUserNum, dtype=bool)))
        np.random.shuffle(testUsers)
        testItemNum = int(self.newWidth*0.5)
        testItems = np.copy(self.X[testUsers, :])
        testItemsRated = ~np.isnan(self.X[testUsers, :])
        self.X[testUsers, :testItemNum] = np.nan
        testItemsRated[:, testItemNum:] = False

        RMSE = 0
        MAE = 0
        eNum = 0
        precision = 0
        recall = 0
        tpNum = 0

        self.ItemBasedPredictionFit(K, threshold, prepareDB=False)
        userNum = 0
        for u in np.argwhere(testUsers).ravel():
            self._ItemBasedRecommendUser(self.newUser[u], recSeen=False)
            testItemsRatedUser = testItemsRated[userNum, :]
            calculatedRatings = self.itemAllUserRatings[testItemsRatedUser]
            realRatings = testItems[userNum, testItemsRatedUser]
            ratingDifference = (calculatedRatings - self.userAve[u]) - realRatings
            relativeCalculatedR = calculatedRatings - self.userAve[u]
            userNum += 1

            if np.sum(testItemsRatedUser) != 0:
                RMSE += sqrt(np.sum(ratingDifference * ratingDifference) / np.sum(testItemsRatedUser))
                MAE += np.sum(np.abs(ratingDifference)) / np.sum(testItemsRatedUser)
                eNum += 1

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

        print("Item based:")
        print("\tRMSE: {:.2f}".format(RMSE / eNum))
        print("\tMAE: {:.2f}".format(MAE / eNum))
        print("\tprecision: {:.2f}".format(precision))
        print("\trecall: {:.2f}".format(recall))
        print("\tF1: {:.2f}".format(2 * (precision * recall) / (recall + precision)))

        RMSE = 0
        MAE = 0
        eNum = 0
        precision = 0
        recall = 0
        tpNum = 0

        self.UserBasedPredictionFit(K, threshold, prepareDB=False)
        userNum = 0
        for u in np.argwhere(testUsers).ravel():
            self._UserBasedRecommendUser(self.newUser[u], recSeen=False)
            testItemsRatedUser = testItemsRated[userNum, :]
            calculatedRatings = self.userAllUserRatings[testItemsRatedUser]
            realRatings = testItems[userNum, testItemsRatedUser]
            ratingDifference = (calculatedRatings - self.userAve[u]) - realRatings
            relativeCalculatedR = calculatedRatings - self.userAve[u]
            userNum += 1

            if np.sum(testItemsRatedUser) != 0:
                RMSE += sqrt(np.sum(ratingDifference * ratingDifference) / np.sum(testItemsRatedUser))
                MAE += np.sum(np.abs(ratingDifference)) / np.sum(testItemsRatedUser)
                eNum += 1

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

        print("Item based:")
        print("\tRMSE: {:.2f}".format(RMSE / eNum))
        print("\tMAE: {:.2f}".format(MAE / eNum))
        print("\tprecision: {:.2f}".format(precision))
        print("\trecall: {:.2f}".format(recall))
        print("\tF1: {:.2f}".format(2 * (precision * recall) / (recall + precision)))

        RMSE = 0
        MAE = 0
        eNum = 0
        precision = 0
        recall = 0
        tpNum = 0

        self.SlopeOnePredictionFit(prepareDB=False)
        userNum = 0
        for u in np.argwhere(testUsers).ravel():
            self._SlopeOneRecommendUser(self.newUser[u], recSeen=False)
            testItemsRatedUser = testItemsRated[userNum, :]
            calculatedRatings = self.s1AllUserRatings[testItemsRatedUser]
            realRatings = testItems[userNum, testItemsRatedUser]
            ratingDifference = (calculatedRatings - self.userAve[u]) - realRatings
            relativeCalculatedR = calculatedRatings - self.userAve[u]
            userNum += 1

            if np.sum(testItemsRatedUser) != 0:
                RMSE += sqrt(np.sum(ratingDifference * ratingDifference) / np.sum(testItemsRatedUser))
                MAE += np.sum(np.abs(ratingDifference)) / np.sum(testItemsRatedUser)
                eNum += 1

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

        print("Item based:")
        print("\tRMSE: {:.2f}".format(RMSE / eNum))
        print("\tMAE: {:.2f}".format(MAE / eNum))
        print("\tprecision: {:.2f}".format(precision))
        print("\trecall: {:.2f}".format(recall))
        print("\tF1: {:.2f}".format(2 * (precision * recall) / (recall + precision)))


