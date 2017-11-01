import pickle
import numpy as np
import time

def loadArray(filename):
    f2 = open(filename, 'r')
    s = pickle.load(f2)
    f2.close()
    return s

def transformXToArray(filename, arrayname):
    start_time = time.time()
    x = np.loadtxt(filename, delimiter=",") # load from text
    x = x.reshape(-1, 64, 64) # reshape
    #save the train set as a numpy array
    print "loaded the train set x in", (time.time() - start_time)
    f = open(arrayname, 'w')
    print "writing to file at ", (time.time())
    pickle.dump(x,f)
    f.close()
    return

#test if loading numpy is faster with test_x
# without = 57s
# with = 11s
transformXToArray('train_x.csv', 'train_x')


