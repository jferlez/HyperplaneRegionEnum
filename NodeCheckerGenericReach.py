import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import cdd
import numpy as np
from posetFastCharm import unflipInt, posHyperplaneSet
import posetFastCharm
from copy import copy
import time

class NodeCheckerGenericReach(Chare):
    
    
    def initialize(self, AbPairs, pt, fixedA, fixedb, localLinearFns, selectorMats):
        self.constraints = None
        self.AbPairs = AbPairs
        self.pt = pt
        self.fixedA = fixedA
        self.fixedb = fixedb
        self.Nstack = len(self.AbPairs[0][0])
        self.nodeIntMask = (2**(self.Nstack+1))-1
        self.localLinearFns = localLinearFns
        self.N = len(self.localLinearFns[0][0])
        self.selectorMatsFull = selectorMats
        self.m = len(self.selectorMatsFull)
        self.selectorSetsFull = [[] for k in range(len(selectorMats))]
        # Convert the matrices to sets of 'used' hyperplanes
        for k in range(len(selectorMats)):
            self.selectorSetsFull[k] = list( \
                    map( \
                        lambda x: frozenset(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
                        self.selectorMatsFull[k] \
                    ) \
                )
        
        self.myWorkList = []
        self.initTime = 0
    
    def setConstraint(self,outA,outb):
        out = 0
        t = time.time()
        self.outA = outA
        self.outb = outb
        self.selectorSets = self.selectorSetsFull[out]
        self.constraints = posetFastCharm.constraints( \
                -1*self.AbPairs[out][0], \
                self.AbPairs[out][1], \
                self.pt, \
                self.fixedA, \
                self.fixedb \
            )
        self.myWorkList = []
        self.initTime += time.time() - t
        return 1
    
    @coro
    def workerInitTime(self, stat_result):
        self.reduce(stat_result, self.initTime, Reducer.sum)
    
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
            
            self.facesList = [-1 for k in range(len(self.myWorkList))]
            self.facesSets = list(map( lambda x: frozenset(posHyperplaneSet(x>>(self.Nstack+1),len(self.constraints.fullConstraints))) , self.myWorkList ))
            print(self.facesSets)
            # now throw away the faces information from the integers
            for k in range(len(self.myWorkList)):
                self.myWorkList[k] = self.myWorkList[k] & self.nodeIntMask
            self.regSets = list(map( lambda x: frozenset(posHyperplaneSet(unflipInt(x,self.constraints.flipMapSet,self.constraints.N),self.constraints.N)) , self.myWorkList ))
            print(self.myWorkList)
            val = True
            for regIdx in range(len(self.regSets)):
                actFns = self.findActiveFunction(regIdx)
                print('Region: ' + str(self.regSets[regIdx]) + '... Active functions: ' + str(actFns))
                # actFns now indexes one local linear function for each output, so this can be used to set up the LP
                val = True
                val = False # Set val to the result of the LP
                if val:
                    break
        else:
            val = False
        # # Set 'val' to the LOGICAL OR of the truth value for each node integer in self.myWorkList
        # val = True if 0 in self.myWorkList else False
        # Leave this line alone
        self.reduce(reduceCallback, val , Reducer.logical_or)


    def findActiveFunction(self,regIdx):
        res = [-1 for ii in range(len(self.selectorMatsFull))]
        minOut = [-1 for ii in range(len(self.selectorMatsFull[0]))]
        for out in range(len(res)):
            for sel in range(len(self.selectorMatsFull[out])):
                selList = list(self.selectorSetsFull[out][sel])
                # print('idx = ' + str(sel) + '; sel before: ' + str(selList))
                for idx in range(len(selList)-1):
                    if selList[idx] == selList[idx+1]:
                        continue
                    # Flip them if they're out of order for this region:
                    if self.compareFns(selList[idx],selList[idx+1],regIdx,out) > 0:
                        temp = selList[idx+1]
                        selList[idx+1] = selList[idx]
                        selList[idx] = temp
                # print('idx = ' + str(sel) + '; sel after: ' + str(selList))
                minOut[sel] = selList[-1]
            # print('minOut before max: ' + str(minOut))
            for idx in range(len(minOut)-1):
                if minOut[idx] == minOut[idx+1]:
                        continue
                if self.compareFns(minOut[idx],minOut[idx+1],regIdx,out) < 0:
                    temp = minOut[idx+1]
                    minOut[idx+1] = minOut[idx]
                    minOut[idx] = temp
            res[out] = minOut[-1]
            # print('minOut after max: ' + str(minOut))
        return res



    def compareFns(self,f1,f2,regIdx,out):
        # Look up depends on having f1 < f2
        sign = 1
        if f2 < f1:
            temp = f2
            f2 = f1
            f1 = temp
            sign = -1
        bigOffsetCnt = f2-f1-1
        bigOffset = 0
        for ii in range(bigOffsetCnt):
            bigOffset += self.N - 1 - ii
        bigOffset = self.m * bigOffset
        smallOffset = out*(self.N - 1 - bigOffsetCnt) + f1
        # print('f1 = ' + str(f1) + '; f2 = ' + str(f2) + '; lookup index = ' + str( bigOffset + smallOffset))
        # print(self.regSets[regIdx])
        if not bigOffset + smallOffset in self.regSets[regIdx]:
            # the f1 - f2 hyperplane is positive, hence f1 >= f2
            return -1 * sign
        else:
            return sign






        # Helper functions:





        # actConstraints = [ \
        #                     self.constraints.fullConstraints[list(facesSets[regIdx]),1:], \
        #                     self.constraints.fullConstraints[list(facesSets[regIdx]),0] \
        #                 ]
        # interiorPoint = findInteriorPoint(*createCDDrep( \
        #         actConstraints[0], \
        #         actConstraints[1].reshape((len(actConstraints[1]),1)) \
        #     ))
        # li = zip(actConstraints[0] @ interiorPoint + actConstraints[1], range(len(actConstraints[1])))
        # li.sort()
        # li = np.sort(np.array(zip(li,range(len(li))))[:,1:3],axis=0)[:,1]
# def createCDDrep(inputConstraintsA, inputConstraintsb):
    
#     inputMat = cdd.Matrix(np.hstack((-1*inputConstraintsb,inputConstraintsA)))
#     inputMat.rep_type = cdd.RepType.INEQUALITY
#     inputPolytope = cdd.Polyhedron(inputMat)
#     vrep = np.array(inputPolytope.get_generators())
#     if len(vrep) == 0:
#         raise ValueError('No vertices for input constraint polyhedron!')
    
#     if np.sum(vrep[:,0]) < len(vrep):
#         raise ValueError('Input constraints do not specify a closed, bounded polyhedron!')
#     inputVrep = vrep[:,1:]

#     return inputMat, inputPolytope, inputVrep

# def findInteriorPoint(inputMat,inputPolytope,inputVrep):
#     activeConstraints = inputPolytope.get_adjacency()[0]
#     # Compare vertices that are non-adjacent to vertex 0, and find the furthest one away
#     d = 0
#     dIdx = -1
#     for k in range(1,len(inputVrep)):
#         if not k in activeConstraints:
#             di = np.linalg.norm(inputVrep[0] - inputVrep[k])
#             if di > d:
#                 d = di
#                 dIdx = k
#     pt = 0.5*(inputVrep[0] + inputVrep[dIdx])
#     pt.reshape( (len(pt),1) )

#     return pt