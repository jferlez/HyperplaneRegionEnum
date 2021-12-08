
# import TLLnet
import numpy as np
import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Channel
import cdd
import cvxopt
import posetFastCharm
from copy import copy,deepcopy
import time
from posetFastCharm import activeHyperplaneSet, unflipInt, posHyperplaneSet, unflipIntFixed
import encapsulateLP
import DistributedHash
import numba as nb
import posetFastCharm_numba

cvxopt.solvers.options['show_progress'] = False

# All hyperplanes assumes to be specified as A x >= b

class PosetNodeTLLVer(DistributedHash.Node):
    def init(self):
        self.constraints, self.selectorSetsFull, self.nodeIntMask, self.out = self.localProxy.getConstraints(ret=True).get()
    def check(self):
        regSet = np.full(self.constraints.N, True, dtype=bool)
        regSet[tuple(self.constraints.flipMapSet),] = np.full(len(self.constraints.flipMapSet),False,dtype=bool)
        regSet[self.nodeBytes,] = np.full(len(self.nodeBytes),False,dtype=bool)
        unflipped = posetFastCharm_numba.is_in_set_idx(self.constraints.flipMapSetNP,list(self.nodeBytes))
        regSet[unflipped] = np.full(len(unflipped),True,dtype=bool)
        regSet = np.nonzero(regSet)[0]

        val = False
        for sSet in self.selectorSetsFull[self.out]:
            if not posetFastCharm_numba.is_non_empty_intersection(regSet,sSet):
                val = True
                break

        return val

# NUMBA jit-able versions of the functions used above; they are slower then the compiled versions
# @nb.cfunc(nb.int64[:](nb.int64[:],nb.int64[:]) )
# def is_in_set_idx(a, b):
#     a = a.ravel()
#     n = len(a)
#     result = np.full(n, 0)
#     set_b = set(b)
#     idx = 0
#     for i in range(n):
#         if a[i] in set_b:
#             result[idx] = i
#             idx += 1
#     return result[0:idx].flatten()
# @nb.cfunc(nb.types.boolean(nb.int64[:],nb.types.Set(nb.int64, reflected=True)) )
# def is_non_empty_intersection(a, set_b):
#     retVal = False
#     a = a.ravel()
#     n = len(a)
#     # set_b = set(b)
#     for i in range(n):
#         if a[i] in set_b:
#             retVal = True
#             return retVal
#     return retVal

class setupCheckerVars(Chare):
    def __init__(self,selectorSetsFull):
        self.selectorSetsFull = selectorSetsFull
        # self.selectorSetsFull = [[] for k in range(len(selectorMats))]
        # # Convert the matrices to sets of 'used' hyperplanes
        # for k in range(len(selectorMats)):
        #     self.selectorSetsFull[k] = list( \
        #             map( \
        #                 lambda x: frozenset(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
        #                 selectorMats[k] \
        #             ) \
        #         )
    def setConstraint(self,constraints, out):
        self.out = out
        # self.selectorSets = self.selectorSetsFull[out]
        self.constraints = constraints
        self.nodeIntMask = [(2**(self.constraints.N+1))-1]
    def getConstraints(self):
        return (self.constraints, self.selectorSetsFull, self.nodeIntMask, self.out)

class TLLHypercubeReach(Chare):
    # @coro
    def __init__(self, localLinearFns, selectorMats, inputConstraints, maxIts, pes, useQuery, useBounding):
        self.maxIts = maxIts
        self.useQuery = useQuery
        self.useBounding = useBounding

        # Transpose local linear function kernels and selector matrices to correct for
        # Keras' multiply-on-the-right convention
        self.localLinearFns = list(map( lambda x: [np.array(x[0]).T, np.array(x[1]).reshape( (len(x[1]),1) )] ,  localLinearFns))
        self.selectorMats = [ list(map( lambda x: np.array(x).T, selectorMats[k] )) for k in range(len(selectorMats)) ]

        self.numOutputs = len(localLinearFns)
        self.n = len(localLinearFns[0][0])
        self.N = len(localLinearFns[0][0][0])
        self.M = len(selectorMats[0])
        self.m = len(localLinearFns)

        self.inputConstraintsA = np.array(inputConstraints[0])
        self.inputConstraintsb = np.array(inputConstraints[1]).reshape( (len(inputConstraints[1]),1) )
        # Create CDD representations for the input constraints
        self.inputMat, self.inputPolytope, self.inputVrep = createCDDrep(self.inputConstraintsA, self.inputConstraintsb)
        # Find a point in the middle of the polyhedron
        self.pt = findInteriorPoint(self.inputMat, self.inputPolytope, self.inputVrep)

        self.selectorSetsFull = [[] for k in range(len(selectorMats))]
        # Convert the matrices to sets of 'used' hyperplanes
        for k in range(len(self.selectorMats)):
            self.selectorSetsFull[k] = list( \
                    map( \
                        lambda x: set(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
                        self.selectorMats[k] \
                    ) \
                )

        self.checkerLocalVars = Group(setupCheckerVars,args=[self.selectorSetsFull])
        charm.awaitCreation(self.checkerLocalVars)


        self.poset = Chare(posetFastCharm.Poset,args=[pes, PosetNodeTLLVer, self.checkerLocalVars],onPE=charm.myPe())
        charm.awaitCreation(self.poset)
        
        stat = self.poset.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, awaitable=True)
        stat.get()

        self.ubCheckerGroup = Group(minGroupFeasibleUB)
        stat = self.ubCheckerGroup.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, self.selectorMats, awaitable=True)
        stat.get()

        self.copyTime = 0
        self.posetTime = 0
        self.workerInitTime = 0
        

    @coro
    def computeReach(self, lbSeed=-1, ubSeed=1, tol=1e-3):
        self.hypercube = np.ones((self.m, 2))
        print('m = ' + str(self.m))
        for out in range(self.m):
            self.hypercube[out,0] = self.searchBound(lbSeed,out=out,lb=True,tol=tol)
            self.hypercube[out,1] = self.searchBound(ubSeed,out=out,lb=False,tol=tol)
        return self.hypercube

    @coro
    def searchBound(self,seedBd,out=0,lb=True,tol=1e-3,verbose=False):
        if out >= self.m:
            raise ValueError('Output ' + str(out) + ' is greater than m = ' + str(self.m))
        # lb2ub = 1
        # if not lb:
        #     lb2ub = -1
        straddle = False
        windLB = -np.inf
        windUB = seedBd
        searchDir = 0
        prevBD = seedBd
        itCnt = self.maxIts
        
        while itCnt > 0:
            bdToCheck = windUB if windLB==-np.inf else 0.5*(windLB + windUB)
            ver = self.verifyLB( bdToCheck, out=out) if lb else self.verifyUB( bdToCheck,out=out)
            
            if verbose:
                print( 'Iteration ' + str(itCnt) +  ': ' + str(bdToCheck) + ' is ' + ('a VALID' if ver else 'an INVALID') + ' lower bound!')
            if windLB == -np.inf:
                # If this is the first pass, decide which way to start looking
                # based on ver:
                if searchDir == 0:
                    searchDir = 1 if ver else -1
                if ver and searchDir > 0:
                    # We're searching right, which means prevBD was a valid lower bound
                    # windUB is a valid lower bound, too, so keep searching right
                    prevBD = windUB
                    searchDir = 1
                    windUB += np.exp(self.maxIts-itCnt)
                elif ver and searchDir < 0:
                    # we were searching left, which means prevBD was NOT a lower bound
                    # Hence, we're now straddling the actual lower bound
                    windLB = windUB
                    windUB = prevBD
                    straddle = True
                elif not ver and searchDir > 0:
                    # We were searching right, which means prevBD WAS a lower bound
                    # Hence, we're now straddling the actual lower bound
                    windLB = prevBD
                    straddle = True
                elif not ver and searchDir < 0:
                    # We're searching left, which means prevBD was not a lower bound
                    # windUB is still not a lower bound, so keep searching left
                    prevBD = windUB
                    searchDir = -1
                    windUB -= np.exp(self.maxIts-itCnt)
            else:
                # Now we know that windLB < actual bound < windUB, and we called the verify function
                # with the midpoint 0.5*(windLB + windUB)
                if ver:
                    windLB = bdToCheck
                else:
                    windUB = bdToCheck
                if np.abs(windUB-windLB) < tol:
                    break
            
            itCnt -= 1
        if not straddle:
            if lb:
                windLB = -np.inf if searchDir < 0 else windUB
            else:
                windUB = np.inf if searchDir > 0 else windUB
        if verbose:
            print('**********    ' + ('verifyLB on LB' if lb else 'verifyUB on UB') + ' processing times:   **********')
            if lb:
                print('Total time required to initialize the new lb problem: ' + str(self.copyTime))
                # collectTimeFut = Future()
                # self.checkerGroup.workerInitTime(collectTimeFut)
                # self.workerInitTime = collectTimeFut.get()
                print('Total time required for region check workers to initialize: ' + str(self.workerInitTime))
                print('Total time required for (partial) poset calculation: ' + str(self.posetTime))
            print('Iterations used: ' + str(self.maxIts - itCnt))
            if not lb:
                print('Total number of LPs used for Upper Bound verification: ' + str(sum(self.ubCheckerGroup.getLPcount(ret=True).get())))
            print('***********************************************************')
        return windLB if lb else windUB

    @coro
    def verifyLB(self,lb, out=0):
        if out >= self.m:
            raise ValueError('Output ' + str(out) + ' is greater than m = ' + str(self.m))
        
        t = time.time()
        
        stat = self.poset.setConstraint(lb, out=out, awaitable=True)
        stat.get()
        self.checkerLocalVars.setConstraint(self.poset.getConstraintsObject(ret=True).get(),out,awaitable=True).get()

        self.copyTime += time.time() - t # Total time across all PEs to set up a new problem

        t = time.time()
        retVal = self.poset.populatePoset(method='fastLP',solver='glpk',findAll=False,useQuery=self.useQuery,useBounding=self.useBounding,ret=True).get() # specify retChannelEndPoint=self.thisProxy to send to a channel as follows
        self.posetTime += time.time() - t

        return retVal
    
    @coro
    def verifyUB(self,ub,out=0):
        if out >= self.m:
            raise ValueError('Output ' + str(out) + ' is greater than m = ' + str(self.m))
        self.ubCheckerGroup.reset(awaitable=True).get()
        self.ubCheckerGroup.checkMinGroup(ub,out)
        minCheckFut = Future()
        self.ubCheckerGroup.collectMinGroupStats(minCheckFut)
        
        retVal = minCheckFut.get()
        print('Upper Bound verifiction used ' + str(sum(self.ubCheckerGroup.getLPcount(ret=True).get())) + ' total LPs.')
        return retVal


class minGroupFeasibleUB(Chare):

    def initialize(self, AbPairs, pt, fixedA, fixedb, selectorMats):
        self.constraints = None
        self.AbPairs = AbPairs
        self.pt = pt
        self.fixedA = fixedA
        self.fixedb = fixedb
        self.N = len(self.AbPairs[0][0])
        self.n = len(self.AbPairs[0][0][0])
        self.selectorMatsFull = selectorMats
        
        self.selectorSetsFull = [[] for k in range(len(selectorMats))]
        # Convert the matrices to sets of 'used' hyperplanes
        for k in range(len(selectorMats)):
            self.selectorSetsFull[k] = list( \
                    map( \
                        lambda x: frozenset(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
                        self.selectorMatsFull[k] \
                    ) \
                )
        
        self.lp = encapsulateLP.encapsulateLP()

        self.selectorIndex = -1
        self.loopback = Channel(self,remote=self.thisProxy[self.thisIndex])
        self.workDone = False
        pes = list(range(charm.numPes()))
        pes.pop(charm.myPe())
        self.otherProxies = [self.thisProxy[k] for k in pes]
        self.tol = 1e-10
    @coro
    def reset(self):
        self.workDone = False
    @coro
    def checkMinGroup(self, ub, out):
        self.status = Future()

        for mySelector in range(charm.myPe(),len(self.selectorSetsFull[out]),charm.numPes()):
            self.loopback.send(1)
            self.loopback.recv()
            if self.workDone:
                break
            n = self.AbPairs[out][0].shape[1]
            # Actually do the feasibility check:
            ubShift = self.AbPairs[out][1][list(self.selectorSetsFull[out][mySelector]),:]
            ubShift = ubShift - ub*np.ones(ubShift.shape)
            bVec = np.vstack([ ubShift , -1*self.fixedb ]).T.flatten()
            selHypers = self.AbPairs[out][0][list(self.selectorSetsFull[out][mySelector]),:]
            status, sol = self.lp.runLP( \
                    np.ones(self.n,dtype=np.float64), \
                    -1*np.vstack([ selHypers, self.fixedA ]), \
                    bVec, \
                    lpopts = {'solver':'glpk'}
                )
            # TO DO: account for intersections that are on the boundary of the input polytope
            if status == 'optimal':
                full = np.vstack([ selHypers, self.fixedA ]) 
                actHypers = np.nonzero(np.abs( full @ sol + bVec) <= self.tol)[0]
                # print('actHypers = ' + str(actHypers))
                # print(sol)
                if len(actHypers) == 0 or np.all((selHypers @ sol + ubShift.flatten()) + self.tol >= 0):
                    for pxy in self.otherProxies:
                        pxy.setDone()
                    self.status.send(True)
                    return
                distinctCount = 1
                solList = [np.array(sol)]
                # print(solList)
                for k in actHypers:
                    # Try to get away from the kth active hyperplane
                    newStatus, newSol = self.lp.runLP( \
                                -full[k,:], \
                                -1*full, \
                                bVec, \
                                lpopts = {'solver':'glpk'}
                            )
                    # print('newSol = ' + str(newSol))
                    # print('Solution difference: '  + str(np.abs(newSol - sol)))
                    if k < len(selHypers) and np.abs(selHypers[k,:] @ newSol + ubShift[k]) <= self.tol \
                        and np.abs(selHypers[k,:] @ solList[-1] + ubShift[k]) <= self.tol \
                        and all([np.linalg.norm(prevSol - newSol) > self.tol for prevSol in solList]):
                        # The feasible set contains a local linear function that is always equal to the upper bound
                        # we're testing, hence the min of this selector set is exactly equal to that upper bound
                        # Hence, this min term does not generate a violation, so we should move on to the next min term
                        print('Degeneracy condition: ' + str(np.abs(selHypers[k,:] @ newSol + ubShift[k])))
                        print('Solution difference: '  + str(np.linalg.norm(newSol - sol)))
                        print('newSol = ' + str(newSol))
                        print('Degenerate upper bound detected')
                        break
                    # print('solList internal: ' + str(solList))
                    if all([np.linalg.norm(prevSol - newSol) > self.tol for prevSol in solList]):
                        # This is a new solution
                        distinctCount += 1
                        solList.append(np.array(newSol))
                        interiorPoint = np.sum(np.hstack(solList),axis=1)/(n+1)
                        # print('Violation condition: ' + str((selHypers @ interiorPoint)+ubShift.flatten()  ) + ' Distinct count ' + str(distinctCount) + ' ' + str(n))
                        # print('LHS = ' + str(selHypers @ np.hstack(solList)) + ' RHS = '  + str(ubShift))
                        # print('Compare LHS = ' + str((np.transpose(selHypers @ np.hstack(solList)) + ubShift.flatten()) + self.tol >= 0) )
                        # print('solList ' + str(solList))
                        # print('selHypers @ interiorPoint = ' + str((selHypers @ interiorPoint) + ubShift.flatten()))
                        if (distinctCount == n + 1 and \
                            np.all(selHypers @ interiorPoint + ubShift.flatten() > self.tol)) or \
                            np.all((selHypers @ newSol + ubShift.flatten()) + self.tol >= 0):
                            # This feasible set has a nonempty interior, so we have a violation
                            # print('sending true')
                            for pxy in self.otherProxies:
                                pxy.setDone()
                            self.status.send(True)
                            return
        self.status.send(False)

    @coro
    def collectMinGroupStats(self, stat_result):
        self.reduce(stat_result, self.status.get(), Reducer.logical_or)
    @coro
    def getLPcount(self):
        return self.lp.lpCount
    @coro
    def setDone(self):
        self.workDone = True



# Helper functions:
def createCDDrep(inputConstraintsA, inputConstraintsb):
    
    inputMat = cdd.Matrix(np.hstack((-1*inputConstraintsb,inputConstraintsA)))
    inputMat.rep_type = cdd.RepType.INEQUALITY
    inputPolytope = cdd.Polyhedron(inputMat)
    vrep = np.array(inputPolytope.get_generators())
    if len(vrep) == 0:
        raise ValueError('No vertices for input constraint polyhedron!')
    
    if np.sum(vrep[:,0]) < len(vrep):
        raise ValueError('Input constraints do not specify a closed, bounded polyhedron!')
    inputVrep = vrep[:,1:]

    return inputMat, inputPolytope, inputVrep

def findInteriorPoint(inputMat,inputPolytope,inputVrep):
    activeConstraints = inputPolytope.get_adjacency()[0]
    # Compare vertices that are non-adjacent to vertex 0, and find the furthest one away
    d = 0
    dIdx = -1
    for k in range(1,len(inputVrep)):
        if not k in activeConstraints:
            di = np.linalg.norm(inputVrep[0] - inputVrep[k])
            if di > d:
                d = di
                dIdx = k
    pt = 0.5*(inputVrep[0] + inputVrep[dIdx])
    pt.reshape( (len(pt),1) )

    return pt