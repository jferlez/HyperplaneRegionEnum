import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import cdd
import numpy as np
from posetFastCharm import unflipInt, posHyperplaneSet, activeHyperplaneSet
import posetFastCharm
from copy import copy
import cvxopt
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
        self.facesMask = (2**(self.Nstack+len(self.fixedA) + 1)-1)
        self.localLinearFns = localLinearFns
        self.N = len(self.localLinearFns[0][0])
        self.n = len(self.localLinearFns[0][0][0])
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
        # Since we're stacking everything the "dummy" problem has only one output
        out = 0
        self.selectorSets = self.selectorSetsFull[out]
        self.constraints = posetFastCharm.constraints( \
                -1*self.AbPairs[out][0], \
                self.AbPairs[out][1], \
                self.pt, \
                self.fixedA, \
                self.fixedb \
            )
        self.myWorkList = []
        self.initTime = 0
    
    def setConstraint(self,outA,outb,eqA=None,eqb=None):
        t = time.time()
        self.outA = np.array(outA)
        self.outb = np.array(outb)
        self.eqA = None if eqA == None else np.array(eqA)
        self.eqb = None if eqb == None else np.array(eqb)
        print(self.eqA)
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
            self.facesSets = list( \
                    map( \
                        lambda x: frozenset( activeHyperplaneSet( self.facesMask & (x>>(self.Nstack+1)),len(self.constraints.fullConstraints) ) ) , \
                        self.myWorkList \
                    ) \
                )
            # print(len(self.facesSets))
            # now throw away the faces information from the integers
            for k in range(len(self.myWorkList)):
                self.myWorkList[k] = self.myWorkList[k] & self.nodeIntMask
            self.regSets = list(map( lambda x: frozenset(posHyperplaneSet(unflipInt(x,self.constraints.flipMapSet,self.constraints.N),self.constraints.N)) , self.myWorkList ))
            val = False
            for regIdx in range(len(self.regSets)):
                actFns = self.findActiveFunction(regIdx)
                # print(actFns)
                # print('Region: ' + str(self.regSets[regIdx]) + '... Active functions: ' + str(actFns))
                # actFns now indexes one local linear function for each output, so this can be used to set up the LP
                # print('constTimesLin = '+ str(self.outA[0,:]))
                for const in range(len(self.outA)):
                    constTimesLin =  self.outA[const,:] @ np.array([ self.localLinearFns[out][0][actFns[out]] for out in range(self.m)])
                    G = -1*np.vstack([ self.AbPairs[0][0] , self.fixedA ])
                    h = np.vstack([ self.AbPairs[0][1] , -self.fixedb ])
                    # print(G)
                    # print(h)
                    for ii in range(len(G)):
                        if ii in self.regSets[regIdx]:
                            G[ii,:] = -G[ii,:]
                            h[ii,:] = -h[ii,:]
                    cvxArgs = [ \
                                cvxopt.matrix( \
                                    constTimesLin.T , \
                                    (self.n,1), \
                                    'd' \
                                ), \
                                cvxopt.matrix( \
                                        G[list(self.facesSets[regIdx]),:], \
                                        (len(self.facesSets[regIdx]), self.n), \
                                        'd' \
                                    ), \
                                cvxopt.matrix( \
                                        h[list(self.facesSets[regIdx]),:], \
                                        (len(self.facesSets[regIdx]),1), \
                                        'd' \
                                    ) \
                            ] + ([] if self.eqA == None else [ cvxopt.matrix(self.eqA,self.eqA.shape,'d'), cvxopt.matrix(self.eqb,self.eqb.shape,'d') ])
                    # print(self.facesSets[regIdx])
                    # print(list(map(lambda x: np.array(x),cvxArgs)))
                    sol = cvxopt.solvers.lp(*cvxArgs)
                    # print(sol)
                    if self.eqA == None and not sol['status'] == 'optimal':
                        raise ValueError('Couldn\'t find a solution to the LP. Something went wrong!')
                    
                    # If the optimum violates the constraint, then we're done
                    print('Optimal solution: ' + str(np.array(sol['x'])))
                    if sol['status'] == 'optimal':
                        constTimesBias = self.outA[const,:] @ np.array([ self.localLinearFns[out][1][actFns[out]] for out in range(self.m)])
                        print('Decision pair: ' + str([(constTimesLin @ np.array(sol['x'])).flatten()[0] + constTimesBias.reshape((1,1))[0,0],  self.outb[const,0]]))
                        if (constTimesLin @ np.array(sol['x'])).reshape((1,1))[0,0] + constTimesBias.reshape((1,1))[0,0] < self.outb[const,0]:
                            val = True
                            break
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
