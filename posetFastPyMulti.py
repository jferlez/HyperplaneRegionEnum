

import cdd
import numpy as np
# import scipy
# import scipy.optimize
# import sys
from copy import copy
import ctypes
# from matplotlib import pyplot as plt

# sys.path.append(r'/Library/gurobi911/mac64/lib/python3.8')
# import gurobipy as gurobi

import multiprocessing.dummy as multiprocessing
from multiprocessing.dummy import Pool, Queue, Lock
# import multiprocessing as multip
import time


class RegionEnumerator:

    def __init__(self,queue, threads=1):
        
        self.resQueue = queue
        self.threads = threads

    def GetRegions(self,constraints):
        self.constraints = constraints

        poset = Poset(self.constraints,self.resQueue)

        t = time.time()

        populatePoset(poset,self.threads)

        return [it.INT for it in list(poset.hashTable.keys())]


class bigInt:
    def __init__(self, i, n):
        self.INT = i
        self.n = n
    def __hash__(self):
        p = 6148914691236517205*(self.INT^(self.INT>>32))
        return 17316035218449499591*(p^(p>>32))
        # p = self.INT
        # return p
    def __eq__(self,other):
        if type(other) == int:
            return self.INT == other
        else:
            return self.INT == other.INT
    def getList(self,flipMap):
        retList = [1 for i in range(self.n)]
        idx = 1
        for i in range(self.n):
            if self.INT & idx > 0:
                retList[i] = -1*flipMap[i]
            else:
                retList[i] = flipMap[i]
            idx = idx << 1
        return retList
    def getInt(self,flipMap):
        retInt = self.INT
        idx = 1
        for i in range(self.n):
            if flipMap[i] < 0:
                retInt = retInt^idx
            idx = idx << 1
        return retInt
    def flipElement(self,k):
        print('2')

def makeList(INT,n):
    retList = [1 for i in range(n)]
    idx = 1
    for i in range(n):
        if INT & idx > 0:
            retList[i] = -1
        idx = idx << 1
    return retList

def convertToList(INT,n):
    retList = [1 for i in range(n)]
    idx = 1
    for i in range(n):
        if INT & idx > 0:
            retList[i] = -1
        idx = idx << 1
    return retList

class constraints:

    def __init__(self, nA, nb, pt, fA=None, fb=None):
        self.flipMapN = np.where((nA @ pt - nb)<0,-1,1)[:,0]
        self.nA = np.diag(self.flipMapN) @ nA
        self.nb = np.diag(self.flipMapN) @ nb

        if (fA is not None) and (fb is not None):
            self.flipMapF = np.where((fA @ pt - fb)<0,-1,1)[:,0]
            self.fA = np.diag(self.flipMapF) @ fA
            self.fb = np.diag(self.flipMapF) @ fb
            self.fullConstraints = np.vstack( ( np.hstack((-1*self.nb,self.nA)), np.hstack((-1*self.fb,self.fA)) ) )

        else:
            self.fA = None
            self.fb = None
            self.flipMapF = None
            self.fullConstraints = np.hstack((-self.nb,self.nA))

class Poset:
    def __init__(self, constraints, resQueue):
        self.constraints = constraints
        self.resQueue = resQueue
        # This number is highly dependent on hardware/LP solver used
        # As a guideline, it should be chosen such that:
        #       (time to get a minimal H-representation) * self.paralellThreshold > 100 mSec
        # (since the overhead associated wtih waiting for pool workers to complete is ~100 mSec)
        self.parallelThreshold = 512

        self.hashTable = {}
        self.levelArray = [[] for i in range(len(self.constraints.nA))]

        self.root = PosetNode(bigInt(0,len(self.constraints.nA)),0, self.constraints)
        self.hashTable[self.root.INTrep] = self.root
        self.levelArray[0].append(self.root)
        self.root.regionLeveled = True        
    
    def shootRay(self, pNode, dir, resQueue):
        print('2')

def initContexts(_successors):
    # fun.H = _H
    global successors
    successors = _successors

def populatePoset(poset,threads):
    startTime = time.time()
    (dimX, dimY) = poset.constraints.fullConstraints.shape
    shared_array_base = multiprocessing.Array('d', poset.constraints.fullConstraints.reshape((-1,1)), lock=False )
    shared_array = np.ctypeslib.as_array(shared_array_base)
    shared_array = shared_array.reshape(dimX, dimY)
    for i in range(dimX):
        for j in range(dimY):
            shared_array[i,j] = poset.constraints.fullConstraints[i,j]
    
    global successors
    successors = [[] for i in range(len(poset.constraints.nA))]

    pool = Pool(processes=threads, initializer=initContexts, initargs=(successors,))
    level = 0
    while level < len(poset.levelArray) and len(poset.levelArray[level]) > 0:
        # If this node hasn't been sent out on the fast queue, do it now
        for idx in range(len(poset.levelArray[level])):
            if not poset.levelArray[level][idx].regionFastQueued:
                while poset.resQueue.full():
                    time.sleep(0.001)
                poset.resQueue.put(poset.levelArray[level][idx].INTrep.getInt(poset.constraints.flipMapN),block=False)
                poset.levelArray[level][idx].regionFastQueued = True
        # Now parallelize the LPs to find the neihboring regions; we won't put them in the poset
        # yet, though
        if len(poset.levelArray[level]) < poset.parallelThreshold:
            parallelCode = False
        else:
            parallelCode = True
            mirrorLevel = [None for i in range(len(poset.levelArray[level]))]
        # print('Pool creation time: ' + str(time.time()-t))
        for idx in range(len(poset.levelArray[level])):
            if parallelCode:
                mirrorLevel[idx] = \
                    pool.apply_async( \
                        processNodeSuccessors, \
                        ( \
                            idx, \
                            poset.levelArray[level][idx].INTrep.INT, \
                            len(poset.constraints.nA), \
                            shared_array \
                        ) \
                    )
            else:
                processNodeSuccessors( \
                            idx, \
                            poset.levelArray[level][idx].INTrep.INT, \
                            len(poset.constraints.nA), \
                            poset.constraints.fullConstraints \
                        )
        t = time.time()

        resStatus = (1 << len(mirrorLevel))-1 if parallelCode else 1

        while resStatus > 0:
            for idx in range(len(poset.levelArray[level])):
                posetNode = poset.levelArray[level][idx]
                if parallelCode:
                    if (not mirrorLevel[idx].ready()) or (resStatus & (1 << idx) <= 0):
                        continue
                posetNode.successors = copy(successors[idx])#mirrorLevel[idx].get()
                # An integer containing the faces for this node is appended to the successors list:
                posetNode.facesInt = posetNode.successors.pop()
                for idx2 in range(len(posetNode.successors)):
                    nodeInt = bigInt(posetNode.successors[idx2],len(poset.constraints.nA))
                    if nodeInt in poset.hashTable:
                        posetNode.successors[idx2] = poset.hashTable[nodeInt]
                        if not posetNode.successors[idx2].regionLeveled:
                            poset.levelArray[level+1].append(posetNode.successors[idx2])
                            posetNode.successors[idx2].regionLeveled = True
                    else:
                        posetNode.successors[idx2] = PosetNode( \
                                bigInt( posetNode.successors[idx2], len(poset.constraints.nA) ), \
                                level + 1, \
                                poset.constraints \
                            )
                        poset.hashTable[posetNode.successors[idx2].INTrep] = posetNode.successors[idx2]
                        poset.levelArray[level+1].append(posetNode.successors[idx2])
                        posetNode.successors[idx2].regionLeveled = True
                resStatus = resStatus - (1 << idx)
        

        level += 1

    # print('Poset creation time: ' + str(time.time()-startTime))
    pool.close()
    pool.join()

def processNodeSuccessors(idxOut,INTrep,N,H2):
    H = copy(H2)
    # global H2
    # H = np.array(H2)
    # H = np.array(processNodeSuccessors.H)
    idx = 1
    for i in range(N):
        if INTrep & idx > 0:
            H[i] = -1*H[i]
        idx = idx << 1
    
    mat = cdd.Matrix(H)
    mat.rep_type = cdd.RepType.INEQUALITY
    ret = mat.canonicalize()
    to_keep = sorted(list(frozenset(range(len(H))) - ret[1]))
    successors[idxOut].clear()
    for i in range(len(to_keep)):
        if to_keep[i] >= N:
            break
        idx = 1 << to_keep[i]
        if idx & INTrep <= 0:
            successors[idxOut].append( \
                    INTrep + idx \
                )
    facesInt = 0
    for k in to_keep:
        facesInt = facesInt + (1 << k)
    successors[idxOut].append(facesInt)
    # return successors


# def processNode(level, pidx):

#     global poset, hashLock
#     posetNode = poset.levelArray[level][pidx]

#     # if level == 2:
#     #     time.sleep(10)
#     # t = time.time()
#     # hashLock.acquire()
#     # # print('processing node: ' + str(posetNode.INTrep.INT))
#     # doRet = False
#     # if posetNode.regionProcessed:
#     #     doRet = True
#     # if not posetNode.regionFastQueued:
#     #     poset.resQueue.put(posetNode.INTrep.getList(poset.constraints.flipMapN),block=False)
#     #     posetNode.regionFastQueued = True
#     # hashLock.release()
#     # if doRet:
#     #     return

    
#     posetNode.identifyFaces()
#     # t = time.time()-t
    
#     # hashLock.acquire()
    
#     # for i in range(len(posetNode.successors)):
#     #     if posetNode.successors[i].INTrep in poset.hashTable:
#     #         posetNode.successors[i] = poset.hashTable[posetNode.successors[i].INTrep]
#     #     else:
#     #         poset.hashTable[posetNode.successors[i].INTrep] = posetNode.successors[i]
#     #         poset.levelArray[posetNode.successors[i].level].append( posetNode.successors[i] )
    
#     # posetNode.regionLeveled = True
#     # posetNode.regionProcessed = True
#     # # print('Total Lock time took : ' + str(t))
#     # hashLock.release()


class PosetNode:

    def __init__(self, INTrep, level, constraints):
        self.INTrep = INTrep
        self.level = level
        self.constraints = constraints
        self.regionFastQueued = False
        self.regionLeveled = False
        self.regionProcessed = False
        self.facesInt = 0
        self.facesList = []
        self.successors = []
    
    # def identifyFaces(self):
    #     H = copy(self.constraints.fullConstraints)
    #     idx = 1
    #     for i in range(len(self.constraints.nA)):
    #         if self.INTrep.INT & idx > 0:
    #             H[i] = -1*H[i]
    #         idx = idx << 1
        
    #     mat = cdd.Matrix(H)
    #     mat.rep_type = cdd.RepType.INEQUALITY
    #     ret = mat.canonicalize()
    #     to_keep = sorted(list(frozenset(range(len(H))) - ret[1]))
    #     for i in range(len(to_keep)):
    #         if to_keep[i] >= len(self.constraints.nA):
    #             break
    #         idx = 1 << to_keep[i]
    #         if idx & self.INTrep.INT <= 0:
    #             self.successors.append( \
    #                     PosetNode( \
    #                         bigInt( self.INTrep.INT + idx, len(self.constraints.nA) ), \
    #                         self.level + 1, \
    #                         self.constraints
    #                     ) \
    #                 )
    
