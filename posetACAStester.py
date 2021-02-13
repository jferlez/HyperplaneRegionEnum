import sys
sys.path.append('./peregriNNtesting/utils')

import posetFastPyMulti
import numpy as np
import time
# from multiprocessing import Process, Queue
from threading import Thread
from multiprocessing.dummy import Queue
from sample_network import split_input_space
import NeuralNetwork
import scipy
import scipy.optimize
import pickle

def f(W,b,pt,fixedConstraintsA,fixedConstraintsb,queue):
    global regionsDirect
    constraints = posetFast.constraints( \
            np.array(W), \
            np.array(b), \
            np.array([pt]).transpose(), \
            np.array(fixedConstraintsA), \
            np.array(fixedConstraintsb) \
        )
    t = time.time()
    regEnum = posetFast.RegionEnumerator(queue)
    # print('Time to intialize RegionEnumerator class is: ' + str(time.time()-t))

    t = time.time()
    regionsDirect = regEnum.GetRegions(constraints)

    # print('Time to fill poset (main thread): ' + str(time.time()-t))
 
    # time.sleep(20)


if __name__ == "__main__":

    network = "peregriNNtesting/models/ACASXU_run2a_2_5_batch_2000.nnet"
    nnet = NeuralNetwork.NeuralNetworkStruct()
    nnet.parse_network(network)
    raw_lower_bounds = np.array([55947.691, -3.141592, -3.141592, 1145, 0]).reshape((-1,1))
    raw_upper_bounds = np.array([62000, 3.141592, 3.141592, 1200, 60]).reshape((-1,1))
    lower_bounds = nnet.normalize_input(raw_lower_bounds)
    upper_bounds = nnet.normalize_input(raw_upper_bounds)
    input_bounds = np.concatenate((lower_bounds,upper_bounds),axis = 1)

    W = nnet.layers[1]['weights'].tolist()
    b = (-nnet.layers[1]['bias']).tolist()
    problems = split_input_space(nnet,input_bounds,128)

    problems.append(input_bounds)

    regCnts = [{} for i in range(len(problems))]


    totTime = time.time()
    fullTime = 0

    for probIdx in range(len(problems)):

        print(' ')
        print('***** Problem ' + str(probIdx) + ' *****')

        # problems=[input_bounds] 
        fixedConstraintsA = []
        fixedConstraintsb = []

        for i in range(len(problems[0])):
            fixedConstraintsA.append([1 if j == i else 0 for j in range(len(problems[0]))])
            fixedConstraintsA.append([-1 if j == i else 0 for j in range(len(problems[0]))])
            fixedConstraintsb.append(problems[probIdx][i][0])
            fixedConstraintsb.append(-problems[probIdx][i][1])

        # fixedConstraintsA = np.array(fixedConstraintsA)
        fixedConstraintsb = np.array([fixedConstraintsb]).transpose().tolist()

        pt = [ (i[0]+i[1])/2 for i in problems[probIdx] ]
        
        resQueue = Queue()

        worker = Thread(target=f, args=(W,b,pt,fixedConstraintsA,fixedConstraintsb,resQueue,))

        worker.start()

        # ***************************************************
        # Use the multiprocessing queue to return results:
        # ***************************************************
        arrivals = [0 for i in range(1000)]
        arrivals[0] = time.time()
        arrIdx = 1
        regions = []
        # Wait for at least one item from the queue, and keep refresh the queue polling loop
        # as long as the worker thread is open
        while len(regions) == 0 or worker.is_alive():
            while not resQueue.empty():
                item = posetFast.makeList(resQueue.get(block=True),len(W))
                regions.append(item)
                arrivals[arrIdx] = time.time()
                arrIdx += 1
                if arrIdx >= len(arrivals):
                    arrivals = arrivals + [0 for i in range(1000)]

        worker.join()
        
        startTime = arrivals[0]
        arrivals = [arrivals[i]-startTime for i in range(1,arrIdx)]


        # # ***************************************************
        # # Direct transfer return:
        # # ***************************************************
        # worker.join()
        # startTime = arrivals[0]
        # regions = [posetFast.makeList(it,len(W)) for it in regionsDirect]

        t = time.time()-startTime
        print('Region computation + transfer time: ' + str(t))
        fullTime = fullTime + t

        regCnts[probIdx]['regions'] = regions
        regCnts[probIdx]['arrivals'] = arrivals

        fullConstraintsA = -1*np.vstack((W,fixedConstraintsA))
        fullConstraintsb = -1*np.vstack((b,fixedConstraintsb))

        validRegions = [False for i in range(len(regions))]
        for rIdx in range(len(regions)):
            flipMat = np.diag( regions[rIdx] + [1 for i in range(len(fixedConstraintsA))] )
            try:
                res = scipy.optimize.linprog( \
                        np.ones(fullConstraintsA.shape[1]), \
                        A_ub=flipMat @ fullConstraintsA, \
                        b_ub=flipMat @ fullConstraintsb, \
                        bounds=(-100,100), \
                        method='interior-point' \
                    )
                validRegions[rIdx] = res.success
            except Exception as e:
                validRegions[rIdx] = False
                print(e)
        regCnts[probIdx]['validRegions'] = validRegions
    
    with open('results_new.p','wb') as fp:
        pickle.dump(regCnts,fp,protocol=pickle.HIGHEST_PROTOCOL)
    
    print('Total time (only region computation): ' + str(fullTime))
    print('Total time (including verification): ' + str(time.time()-totTime))
        
