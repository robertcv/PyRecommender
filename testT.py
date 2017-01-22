import numpy as np
import threading

class thr:
    def __init__(self, vec1, num):
        self.vec1 = vec1
        self.num = num
        self.odg = np.zeros(num)

    def thred(self, vec2):
        threads = []
        for i in range(self.num):
            t = threading.Thread(target=self.worker, args=(i,vec2, ))
            threads.append(t)
            t.start()
        print("start")
        for i in range(self.num):
            threads[i].join()
        print("end")


    def worker(self, i, vec2):
        self.odg[i] = np.dot(self.vec1, vec2)
        print(str(i) + " - " + str(self.odg[i]))

Thr = thr(np.random.randint(1000000, size=500000000), 8)
Thr.thred(np.random.randint(1000000, size=500000000))

