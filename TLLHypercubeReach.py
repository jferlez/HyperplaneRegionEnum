
import TLLnet
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


cvxopt.solvers.options['show_progress'] = False

# All hyperplanes assumes to be specified as A x >= b

class PosetNodeTLLVer(DistributedHash.Node):
    def init(self):
        self.constraints, self.selectorSetsFull, self.nodeIntMask, self.out = self.localProxy.getConstraints(ret=True).get()
    def check(self):
        regSet = frozenset(
                activeHyperplaneSet(\
                    unflipIntFixed(self.nodeInt & self.nodeIntMask[0],self.constraints.flipMapSet,self.constraints.N), \
                    self.constraints.N \
                ) \
            )
        val = False
        for sSet in self.selectorSetsFull[self.out]:
            if len(sSet & regSet) == 0:
                val = True
                break

        return val

class setupCheckerVars(Chare):
    def __init__(self,selectorMats):
        self.selectorSetsFull = [[] for k in range(len(selectorMats))]
        # Convert the matrices to sets of 'used' hyperplanes
        for k in range(len(selectorMats)):
            self.selectorSetsFull[k] = list( \
                    map( \
                        lambda x: frozenset(np.flatnonzero(np.count_nonzero(x, axis=0)>0)), \
                        selectorMats[k] \
                    ) \
                )
    def setConstraint(self,constraints, out):
        self.out = out
        # self.selectorSets = self.selectorSetsFull[out]
        self.constraints = constraints
        self.nodeIntMask = [(2**(self.constraints.N+1))-1]
    def getConstraints(self):
        return (self.constraints, self.selectorSetsFull, self.nodeIntMask, self.out)

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

        self.inputConstraintsA = np.array(inputConstraints[0])
        self.inputConstraintsb = np.array(inputConstraints[1]).reshape( (len(inputConstraints[1]),1) )
        # Create CDD representations for the input constraints
        self.inputMat, self.inputPolytope, self.inputVrep = createCDDrep(self.inputConstraintsA, self.inputConstraintsb)
        # Find a point in the middle of the polyhedron
        self.pt = findInteriorPoint(self.inputMat, self.inputPolytope, self.inputVrep)

        self.checkerLocalVars = Group(setupCheckerVars,args=[self.selectorMats])
        charm.awaitCreation(self.checkerLocalVars)


        self.poset = Chare(posetFastCharm.Poset,args=[{'poset':[(0,4,1)],'hash':[(0,4,1)]}, PosetNodeTLLVer, self.checkerLocalVars],onPE=charm.myPe())
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
        retVal = self.poset.populatePoset(method='fastLP',solver='glpk',findAll=False,ret=True).get() # specify retChannelEndPoint=self.thisProxy to send to a channel as follows
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
            # Actually do the feasibility check:
            ubShift = self.AbPairs[out][1][list(self.selectorSetsFull[out][mySelector]),:]
            ubShift = ubShift - ub*np.ones(ubShift.shape)
            bVec = np.vstack([ ubShift , -1*self.fixedb ]).T.flatten()
            status, sol = self.lp.runLP( \
                    np.ones(self.n,dtype=np.float64), \
                    -1*np.vstack([ self.AbPairs[out][0][list(self.selectorSetsFull[out][mySelector]),:], self.fixedA ]), \
                    bVec, \
                    lpopts = {'solver':'glpk'}
                )
            # TO DO: account for intersections that are on the boundary of the input polytope
            if status == 'optimal':
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