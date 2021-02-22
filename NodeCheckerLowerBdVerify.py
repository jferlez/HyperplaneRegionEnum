import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import cdd
import numpy as np
from posetFastCharm import unflipInt, posHyperplaneSet
from copy import copy

class NodeCheckerLowerBdVerify(Chare):
    # Not strictlly necessary
    # @coro
    def __init__(self, constraints, selectorMats):
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