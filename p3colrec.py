# simple collaborative recommender
# with utilities to get recommendations and interesting stats directly from python interpreter
# written for Decision Systems (OS) classes
# by Aleksander Sadikov, November, December, and January 2011

from math import sqrt
#from exceptions import Exception
from random import randint

class ColRecommender:
    def __init__(self, fileName, thr = 0.5):
        # initializes recommender from the database
        # and calculates average ratings for items and users
        self.fileName = fileName
        self.thr = thr
        self.kThr = 3   # min neighbourhood size (see switch hybrid)
        try:
            self.parseDatabase()
            self.computeAverages()
        except:
            raise Exception('Error in database.')

    def parseDatabase(self):
        self.items, self.ratings = [], {}
        mode = 'none'
        data = open(self.fileName, 'r').readlines()
        for line in data:
            ln = line.strip()
            if not ln or ln[0] == '%': continue    # empty line or comment
            if ln == '[items]':
                # switch to parsing item data
                mode = 'items'
                continue
            if ln == '[users]':
                # switch to parsing user/rating data
                mode = 'users'
                iCount = len(self.items)
                continue
            if mode == 'items':
                self.items.append(ln)
            elif mode == 'users':
                ln = ln.split(',')
                # missing ratings (?) are represented with zeros, which are logically False
                self.ratings[ln[0]] = list(map(lambda x: int(x) if x.strip() != '?' else 0, ln[1:]))
                if len(self.ratings[ln[0]]) != iCount:    # check DB consistency
                    print("User %s has invalid number of ratings (%d)." % (ln[0], len(self.ratings[ln[0]])))
            else:
                print('Strange line in database:')
                print(line)

    def computeAverages(self):
        # computes average scores the user gives (for all users) and average scores of items
        # all users need to have at least one valid rating or a database error is raised
        self.avgs = {}
        for user, ratings in self.ratings.items():
            self.avgs[user] = float(sum(ratings)) / sum(list(map(bool, ratings)))

        # create a dictionary of item ratings (transposing the user ratings matrix)
        self.iratings = dict(list(map(lambda x: (x, []), self.items)))
        for ratings in self.ratings.values():
            for ix, rating in enumerate(ratings):
                self.iratings[self.items[ix]].append(rating)

        # compute item average ratings
        self.iavgs = {}
        for item, ratings in self.iratings.items():
            if not sum(list(map(bool, ratings))):
                print(item)
                print(ratings)
            self.iavgs[item] = float(sum(ratings)) / sum(list(map(bool, ratings)))
            
    def printDatabase(self):
        print('Items:')
        for item in self.items:
            print("  %s" % item)
        print()
        print('Users & ratings:')
        for user, ratings in self.ratings.items():
            print("%20s:  %s" % (user, ratings))

    def printItems(self):
        print('Items:')
        for ix, item in enumerate(self.items):
            print(" %2d:  %.2f  %s" % (ix, self.iavgs[item], item))

    def userSimilarity(self, user1, user2):
        # computes similarity between two given users using Pearson's coefficient
        sum1, sum2, sumMixed = 0, 0, 0
        for item in range(len(self.items)):
            if not self.ratings[user1][item] or not self.ratings[user2][item]: continue
            rap = self.ratings[user1][item] - self.avgs[user1]
            rbp = self.ratings[user2][item] - self.avgs[user2]
            sum1 += rap ** 2
            sum2 += rbp ** 2
            sumMixed += rap * rbp
        if sum1 == 0 or sum2 == 0: return 0     # special case: similarity could not be computed
        return sumMixed / (sqrt(sum1) * sqrt(sum2))

    def itemSimilarity(self, item1, item2):
        # computes similarity between two given items using adjusted cosine measure
        sum1, sum2, sumMixed = 0, 0, 0
        for user in self.ratings.keys():
            if not self.ratings[user][item1] or not self.ratings[user][item2]: continue
            rua = self.ratings[user][item1] - self.avgs[user]
            rub = self.ratings[user][item2] - self.avgs[user]
            sum1 += rua ** 2
            sum2 += rub ** 2
            sumMixed += rua * rub
        if sum1 == 0 or sum2 == 0: return 0     # special case: similarity could not be computed
        return sumMixed / (sqrt(sum1) * sqrt(sum2))
        
    def recommendItem(self, activeUser, item, explanation = False):
        # calculates the predicted rating the user would give the given item
        # returns either recommendation or 0 (False) if no similar users who rated this item found
        # uses user-based NN recommendation technique
        votes, evotes = [], []
        for user, ratings in self.ratings.items():
            if not ratings[item]: continue      # this user has not rated the given item
            similarity = self.userSimilarity(activeUser, user)
            if similarity >= self.thr:
                votes.append( (ratings[item] - self.avgs[user], similarity) )
                evotes.append( (ratings[item], similarity) )
        if not votes: return 0      # no similar users who rated this item found
        if not explanation:
            # return rating and neighbourhood size
            return self.avgs[activeUser] + sum(map(lambda x: x[0]*x[1], votes)) \
                   / sum(map(lambda x: x[1], votes)), len(votes)
        else:
            # return rating, neighbourhood size, and an explanation
            verysim = ', '.join([str(v) for v, sim in evotes if sim >= 0.7])
            sim = ', '.join([str(v) for v, sim in evotes if sim < 0.7 and sim >= self.thr])
            if verysim:
                expl = 'Some very similar users to you rated this item as: %s.\n' % verysim
            else:
                expl = ''
            if sim:
                expl += 'Some similar users to you rated this item as: %s.\n' % sim
            return self.avgs[activeUser] + sum(map(lambda x: x[0]*x[1], votes)) \
                   / sum(map(lambda x: x[1], votes)), len(votes), expl       

    def recommendItemIB(self, user, activeItem):
        # calculates the predicted rating the user would give the given item
        # returns either recommendation or 0 (False) if no similar users who rated this item found
        # uses item-based NN recommendation technique
        votes = []
        for item in range(len(self.items)):
            if not self.ratings[user][item]: continue   # this item was not rated by the user
            similarity = self.itemSimilarity(activeItem, item)
            if similarity >= self.thr:
                votes.append( (self.ratings[user][item], similarity) )
        if not votes: return 0      # no similar items rated by the active user were found
        return sum(map(lambda x: x[0]*x[1], votes)) / sum(map(lambda x: x[1], votes)), len(votes)

    def recommendItemS1(self, actUser, actItem):
        # calculates the predicted rating the user would give the given item
        # returns either recommendation or 0 (False) if no appropriate data found
        # uses Slope One (S1) technique
        score, totpairs = 0.0, 0
        for item in range(len(self.items)):
            if not self.ratings[actUser][item]: continue    # no rating for this item by active user
            diffSum, npairs = 0.0, 0
            for user in self.ratings.keys():
                if not self.ratings[user][item] or not self.ratings[user][actItem]: continue
                diffSum += self.ratings[user][actItem] - self.ratings[user][item]
                npairs += 1
            if not npairs: continue
            score += npairs * (diffSum / npairs + self.ratings[actUser][item])
            totpairs += npairs
        if not totpairs: return 0   # no common pairs found
        return score / totpairs, totpairs

    def recommendItemConst(self, user, item):
        # stupid recommender
        return (3.3, None)

    def recommendItemRandom(self, user, item):
        # the stupidest recommender
        return (randint(1,5), None)

    def recommendAllItems(self, user):
        # computes and prints ratings (recommendations) for user's unrated items
        for item, itemName in enumerate(self.items):
            if self.ratings[user][item]: continue   # user has rated this item
            recU  = self.recommendItem(user, item)
            #recI  = self.recommendItemIB(user, item)
            #recS1 = self.recommendItemS1(user, item)
            #recSp = self.recommendItemSparse(user, item)
            rcU  = '----------  ' if not recU else '%.2f (%3d)  ' % recU
            #rcI  = '----------  ' if not recI else '%.2f (%3d)  ' % recI
            #rcS1 = '----------  ' if not recS1 else '%.2f (%3d)  ' % recS1
            #rcSp = '----------  ' if not recSp else '%.2f (%3d)  ' % recSp
            #print '%s%s%s%s%s' % (rcU, rcI, rcS1, rcSp, itemName)
            #print('%s%s%s%s' % (rcU, rcI, rcS1, itemName))
            print('%s%s' % (rcU, itemName))

    def testLOO(self, recMethod, verbose=True):
        # tests the given recommendation method using leave-one-out procedure
        mae, rmse, N = 0.0, 0.0, 0
        for user in self.ratings.keys():
            for item in range(len(self.items)):
                if not self.ratings[user][item]: continue   # no rating for this item
                rating, self.ratings[user][item] = self.ratings[user][item], 0  # hide test rating
                predRating = recMethod(user, item)
                self.ratings[user][item] = rating   # restore hidden rating
                if not predRating: continue
                predRating = predRating[0]          # take only prediction, discard reliability data
                mae += abs(predRating - rating)
                rmse += (predRating - rating) ** 2
                N += 1
        if verbose:
            print(' MAE: %.3f' % (mae / N))
            print('RMSE: %.3f' % sqrt(rmse / N))
            print('Number of comparisons: %d' % N)
        return mae / N, sqrt(rmse / N), N

    def getUserSimilarities(self):
        # returns similarities between all users sorted from most to least similar
        sims = []
        for user1 in self.ratings.keys():
            for user2 in self.ratings.keys():
                if user1 >= user2: continue
                sims.append((self.userSimilarity(user1, user2), user1, user2))
        return sorted(sims, reverse = True)

    def prepareAssocRuleDataFile(self):
        # prepares data for mining association rules in Orange format
        # it has the same filename as the original database with .tab extension
        # the transformation is made into two-value (like, dislike) rating system
        # this file is prepared for finding associations between items
        f = open(self.fileName[:-4] + '.tab', 'wt')
        f.write('\t'.join(['user'] + self.items) + '\n')
        f.write('\t'.join(['d'] * (len(self.items) + 1)) + '\n')
        f.write('\t'.join(['meta'] + [''] * len(self.items)) + '\n')
        for user, ratings in self.ratings.items():
            f.write('\t'.join([user] + map(lambda x: '?' if not x else
                                           ('1' if x >= self.avgs[user] else '0'), ratings)) + '\n')
        f.close()
        
    def prepareAssocRuleUserDataFile(self):
        # prepares data for mining association rules in Orange format
        # it has the same filename as the original database with .tab extension
        # the transformation is made into two-value (like, dislike) rating system
        # this file is prepared for finding associations between users
        # TRANSFORMATION INTO TWO-VALUE RATING SYSTEM IS CURRENTLY INVALID
        # (averages are not completely correct)
        users = self.ratings.keys()
        f = open(self.fileName[:-4] + '_user.tab', 'wt')
        f.write('\t'.join(['item'] + users) + '\n')
        f.write('\t'.join(['d'] * (len(users) + 1)) + '\n')
        f.write('\t'.join(['meta'] + [''] * len(users)) + '\n')
        for item, ratings in self.iratings.items():
            f.write('\t'.join([item] + map(lambda x: '?' if not x else
                                           ('1' if x >= self.iavgs[item] else '0'), ratings)) + '\n')
        f.close()

    def recommendItemToAllUsers(self, item, recMethod):
        # gets some stats
        # does not consider attacking profiles into the stats
        recs = {}
        pos, all = 0, 0
        for user in self.ratings.keys():
            if 'attack' in user: continue
            all += 1
            if recMethod(user, item):
                predRating = recMethod(user, item)[0]
                if predRating >= self.avgs[user]:
                    pos += 1
                    recs[user] = 'yes'
                else:
                    recs[user] = 'no'

        print('Average rating for %s is %.2f.' % (self.items[item], self.iavgs[self.items[item]]))
        print('Recommend to %d out of %d users.' % (pos, all))
        return recs
        










            
