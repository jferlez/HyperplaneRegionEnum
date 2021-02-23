
import TLLnet
import numpy as np
import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Channel
import scipy.optimize
import cdd
import posetFastCharm
import NodeCheckerLowerBdVerify
from copy import copy,deepcopy

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

        # stat = self.checkerGroup.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, self.selectorMats, awaitable=True)
        # stat.get()

        self.poset = Chare(posetFastCharm.Poset,args=[self.checkerGroup,[],False,None],onPE=charm.myPe())
        
        stat = self.poset.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, awaitable=True)
        stat.get()

        


    def computeReach(self, lbSeed=-1, ubSeed=1, tol=1e-3):
        pass

    @coro
    def searchBound(self,seedBd,out=0,lb=True,tol=1e-3):
        stat = self.checkerGroup.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, self.selectorMats, awaitable=True)
        stat.get()

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
            print(itCnt)
            print([windLB,windUB])
            bdToCheck = windUB if windLB==-np.inf else 0.5*(windLB + windUB)
            ver = self.verifyLB( bdToCheck, itCnt, out=out) if lb else not self.verifyUB( bdToCheck,out=out)
            # ver = verF.get()
            # ver = bdToCheck < 0.5 if lb else not (bdToCheck > 6.214587)
            print('Found ver = ' + str(ver) + ' on iteration ' + str(itCnt))
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
        print('Iterations used: ' + str(self.maxIts - itCnt))
        return windLB if lb else windUB

    @coro
    def verifyLB(self,lb, it, out=0):
        # print('my PE: ' + str(charm.myPe()))
        # constraints = posetFastCharm.constraints( \
        #     -1*self.localLinearFns[out][0], \
        #     self.localLinearFns[out][1] - lb*np.ones((self.N,1)), \
        #     self.pt, \
        #     self.inputConstraintsA, \
        #     self.inputConstraintsb \
        # )
        # print(constraints.fullConstraints)
        
        print('Verifying lower bound of ' + str(lb))
        # stat = self.checkerGroup.initializeFromConstraints(constraints, self.selectorMats[out],awaitable=True)
        stat = self.checkerGroup.setConstraint(lb, awaitable=True)
        stat.get()
        # charm.sleep(1)
        # print(stat)
        # stat.get()        
        # self.poset.initializeFromConstraintObject(constraints)
        stat = self.poset.setConstraint(lb, awaitable=True)
        stat.get()
        # self.poset.seeConstraints()
        # charm.awaitCreation(self.checkerGroup,poset)
        # charm.sleep(1)
        
        checkFut = Future()
        self.poset.populatePoset(checkNodesFuture=checkFut) # specify retChannelEndPoint=self.thisProxy to send to a channel as follows
        # retChannel = Channel(self, remote=poset)
        retVal = checkFut.get()
        
        # print('*** Running verifyLB on LB = ' + str(lb) + ' ; on itCnt = ' + str(it))
        return not retVal
    
    def verifyUB(self,ub,out=0):
        pass





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