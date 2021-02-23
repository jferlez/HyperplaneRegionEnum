import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import cdd
import numpy as np
from posetFastCharm import unflipInt, posHyperplaneSet
import posetFastCharm
from copy import copy

class NodeCheckerLowerBdVerify(Chare):
    # Not strictlly necessary
    # @coro
    def initializeFromConstraints(self, constraints, selectorMats):
        self.constraints = constraints
        self.selectorMats = selectorMats
        # Converve the matrices to sets of 'used' hyperplanes
        self.selectorSets = list( \
                map( \
                    lambda x: frozenset(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
                    self.selectorMats \
                ) \
            )
        # print(self.selectorSets)
        self.myWorkList = []
    
    def initialize(self, AbPairs, pt, fixedA, fixedb, selectorMats):
        self.constraints = None
        self.AbPairs = AbPairs
        self.pt = pt
        self.fixedA = fixedA
        self.fixedb = fixedb
        self.N = len(self.AbPairs[0][0])
        self.selectorMatsFull = selectorMats
        # self.selectorMats = selectorMats
        self.selectorSetsFull = [[] for k in range(len(selectorMats))]
        # Converve the matrices to sets of 'used' hyperplanes
        for k in range(len(selectorMats)):
            self.selectorSetsFull[k] = list( \
                    map( \
                        lambda x: frozenset(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
                        self.selectorMatsFull[k] \
                    ) \
                )
        # print(self.selectorSets)
        self.myWorkList = []
    # @coro
    def setConstraint(self,lb,out=0):
        self.selectorSets = self.selectorSetsFull[out]
        self.constraints = posetFastCharm.constraints( \
                -1*self.AbPairs[out][0], \
                self.AbPairs[out][1] - lb*np.ones((self.N,1)), \
                self.pt, \
                self.fixedA, \
                self.fixedb \
            )
        self.lb = lb
        self.myWorkList = []
        return 1
    
    @coro
    def initList(self, myWorkList):
        self.status = Future()
        self.myWorkList = myWorkList
        self.status.send(1)

    @coro
    def collectXferStats(self, stat_result):
        self.reduce(stat_result, self.status.get(), Reducer.sum)

    @coro
    def check(self, reduceCallback):
        # print('Trying to verify LB = ' + str(self.lb) + ' using the following workList:')
        # print(self.myWorkList)
        # print('****************************************')
        # print('Using constraints object : ')
        # print(self.constraints.fullConstraints)
        # print('****************************************')
        if len(self.myWorkList) > 0:
            val = True
            for regSet in map( lambda x: frozenset(posHyperplaneSet(unflipInt(x,self.constraints.flipMapSet),self.constraints.N)) , self.myWorkList ):
                val = True
                for sSet in self.selectorSets:
                    if len(sSet & regSet) == 0:
                        val = False
                        break
                if val:
                    break
        else:
            val = False
        # # Set 'val' to the LOGICAL OR of the truth value for each node integer in self.myWorkList
        # val = True if 0 in self.myWorkList else False
        # Leave this line alone
        # print('NodeChecker on ' + str(charm.myPe()) + ' returning ' + str(val) + ' for its workgroup')
        self.reduce(reduceCallback, val , Reducer.logical_or)