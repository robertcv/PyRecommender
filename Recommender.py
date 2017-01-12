import numpy as np

class Recommender(object):
    def __init__(self, predictor=None):
        self.predictor = predictor


    def parse_moviebase(self, file_name):
        data = open(file_name, 'rt', encoding='utf-8')
        self.items = []
        self.users = []
        self.user_ratings = []
        mode = 'none'
        for line in data:
            ln = line.strip()
            if not ln or ln[0] == '%': continue  # empty line or comment
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
                if len(ln) != iCount + 1:  # check DB consistency
                    print("User %s has invalid number of ratings (%d)." % (ln[0], len(self.ratings[ln[0]])))
                self.user_ratings.append([])
                self.users.append(ln[0])
                for v in ln[1:]:
                    v = v.strip()
                    if v == '?':
                        self.user_ratings[-1].append(0)
                    else:
                        self.user_ratings[-1].append(float(v))
            else:
                print('Strange line in database:')
                print(line)
        # convert to numpy
        self.user_ratings = np.array(self.user_ratings, dtype=np.int8)

    def learn(self):
        self.predictor.fit(self.user_ratings)

    def recommend(self, user, n=10, rec_seen=True):
        user_id = self.users.index(user)
        rec_tup = np.array(list(zip(self.items, self.predictor.predict(user_id))))
        rec_bool = np.logical_not(np.array(self.user_ratings[user_id], dtype=bool)) | rec_seen
        rec = rec_tup[rec_bool]
        return rec[rec[:,1].argsort()][-n:]


    def __str__(self):
        return str(self.items) + "\n" + str(self.users) + "\n" + str(self.user_ratings)