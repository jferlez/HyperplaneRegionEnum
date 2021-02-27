
import TLLnet
import numpy as np
import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Channel
import scipy.optimize
import cdd
import cvxopt
import posetFastCharm
import NodeCheckerLowerBdVerify
from copy import copy,deepcopy
import time

cvxopt.solvers.options['show_progress'] = False

# All hyperplanes assumes to be specified as A x >= b


class TLLHypercubeReach(Chare):
    # @coro
    def __init__(self, localLinearFns, selectorMats, inputConstraints, maxIts):
        self.maxIts = maxIts

        # Transpose local linear function kernels and selector matrices to correct for
        # Keras' multiply-on-the-right convention
        self.localLinearFns = list(map( lambda x: [np.array(x[0]).T, np.array(x[1]).reshape( (len(x[1]),1) )] ,  localLinearFns))
        self.selectorMats = [ list(map( lambda x: np.array(x).T, selectorMats[k] )) for k in range(len(selectorMats)) ]

        self.numOutputs = len(localLinearFns)
        self.n = len(localLinearFns[0][0])
        self.N = len(localLinearFns[0][0][0])
        self.M = len(selectorMats[0])
        self.m = len(localLinearFns)

        self.inputConstraintsA = np.array(inputConstraints[0]).T
        self.inputConstraintsb = np.array(inputConstraints[1]).reshape( (len(inputConstraints[1]),1) )
        # Create CDD representations for the input constraints
        self.inputMat, self.inputPolytope, self.inputVrep = createCDDrep(self.inputConstraintsA, self.inputConstraintsb)
        # Find a point in the middle of the polyhedron
        self.pt = findInteriorPoint(self.inputMat, self.inputPolytope, self.inputVrep)

        self.checkerGroup = Group(NodeCheckerLowerBdVerify.NodeCheckerLowerBdVerify)

        stat = self.checkerGroup.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, self.selectorMats, awaitable=True)
        stat.get()

        self.poset = Chare(posetFastCharm.Poset,args=[self.checkerGroup,[],False,None],onPE=charm.myPe())
        
        stat = self.poset.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, awaitable=True)
        stat.get()

        self.ubCheckerGroup = Group(minGroupFeasibleUB)
        stat = self.ubCheckerGroup.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, self.selectorMats, awaitable=True)
        stat.get()

        self.copyTime = 0
        self.posetTime = 0
        self.workerInitTime = 0
        


    def computeReach(self, lbSeed=-1, ubSeed=1, tol=1e-3):
        self.hypercube = np.ones(self.m, 2)
        for out in range(self.m):
            self.hypercube[out,0] = self.searchBound(lbSeed,out=out,lb=True,tol=tol)
            self.hypercube[out,1] = self.searchBound(ubSeed,out=out,lb=False,tol=tol)
        return self.hypercube

    @coro
    def searchBound(self,seedBd,out=0,lb=True,tol=1e-3,verbose=False):
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
            print('**********    verifyLB on LB processing times:   **********')
            print('Total time required to initialize the new lb problem: ' + str(self.copyTime))
            collectTimeFut = Future()
            self.checkerGroup.workerInitTime(collectTimeFut)
            self.workerInitTime = collectTimeFut.get()
            print('Total time required for region check workers to initialize: ' + str(self.workerInitTime))
            print('Total time required for (partial) poset calculation: ' + str(self.posetTime))
            print('Iterations used: ' + str(self.maxIts - itCnt))
            print('***********************************************************')
        return windLB if lb else windUB

    @coro
    def verifyLB(self,lb, out=0):
        
        # Alternate method of resetting poset problem for the new lower bound:
        # ** BETTER DOUBLE CHECK THESE METHODS -- THEY MAY BE OUT OF DATE **

        # constraints = posetFastCharm.constraints( \
        #     -1*self.localLinearFns[out][0], \
        #     self.localLinearFns[out][1] - lb*np.ones((self.N,1)), \
        #     self.pt, \
        #     self.inputConstraintsA, \
        #     self.inputConstraintsb \
        # )
        # stat = self.checkerGroup.initializeFromConstraints(constraints, self.selectorMats[out],awaitable=True)
        # stat.get()
        # stat = self.poset.initializeFromConstraintObject(constraints)
        # stat.get()
        
        t = time.time()
        
        stat = self.checkerGroup.setConstraint(lb, awaitable=True)
        stat.get()
        stat = self.poset.setConstraint(lb, awaitable=True)
        stat.get()

        self.copyTime += time.time() - t # Total time across all PEs to set up a new problem

        t = time.time()
        checkFut = Future()
        self.poset.populatePoset(checkNodesFuture=checkFut) # specify retChannelEndPoint=self.thisProxy to send to a channel as follows
        retVal = checkFut.get()
        self.posetTime += time.time() - t

        return not retVal
    
    @coro
    def verifyUB(self,ub,out=0):
        
        for ii in range(0, len(self.selectorMats[out]), charm.numPes()):
            for k in range(charm.numPes()):
                if ii+k < len(self.selectorMats[out]):
                    self.ubCheckerGroup[k].checkMinGroup(ub,ii+k,out)
                else:
                    self.ubCheckerGroup[k].checkMinGroup(ub,-1, out)
            minCheckFut = Future()
            self.ubCheckerGroup.collectMinGroupStats(minCheckFut)
            val = minCheckFut.get()
            if val:
                return True
        return False



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
        
        self.selectorIndex = -1
    
    @coro
    def checkMinGroup(self, ub, mySelector, out):
        self.status = Future()
        if mySelector < 0:
            self.status.send(False)
            return
        self.selectorIndex = mySelector
        # Actually do the feasibility check:
        ubShift = self.AbPairs[out][1][list(self.selectorSetsFull[out][mySelector]),:]
        ubShift = ubShift - ub*np.ones(ubShift.shape)
        bVec = np.vstack([ ubShift , -1*self.fixedb ]).T.flatten()
        sol = cvxopt.solvers.lp( \
                cvxopt.matrix(np.ones(self.n),(self.n,1),'d'), \
                cvxopt.matrix( \
                        -1*np.vstack([ self.AbPairs[out][0][list(self.selectorSetsFull[out][mySelector]),:], self.fixedA ]),
                        (len(list(self.selectorSetsFull[out][mySelector]))+len(self.fixedA), self.n), \
                        'd' \
                    ), \
                cvxopt.matrix( \
                        bVec,
                        (len(bVec),1),
                        'd' \
                    ) \
            )
        # TO DO: account for intersections that are on the boundary of the input polytope
        if sol['status'] == 'optimal':
            self.status.send(True)
        else:
            self.status.send(False)

    @coro
    def collectMinGroupStats(self, stat_result):
        self.reduce(stat_result, self.status.get(), Reducer.logical_or)



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