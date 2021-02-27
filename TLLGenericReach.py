import TLLnet
import numpy as np
import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Channel
import scipy.optimize
import cdd
import cvxopt
import posetFastCharm
import NodeCheckerGenericReach
from copy import copy,deepcopy
import time

cvxopt.solvers.options['show_progress'] = False

# All hyperplanes assumes to be specified as A x >= b


class TLLGenericReach(Chare):
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

        # In the generic verifier, we need to consider the intersections between PAIRS of local linear functions,
        # so we will generate these 'auxilliary' hyperplanes
        # THIS ASSUMES THAT N is the SAME for each output/min group!!!
        self.stackedLocalLinearFns = [[ \
                np.vstack([self.localLinearFns[out][0] for out in range(len(self.localLinearFns))]),
                np.vstack([self.localLinearFns[out][1] for out in range(len(self.localLinearFns))]),
            ]]
        
        self.pairedLocalLinearFns = [[0,1]]
        for kb in [0,1]:
            self.pairedLocalLinearFns[0][kb] = \
                np.vstack( \
                    [ \
                        (self.stackedLocalLinearFns[0][kb] - np.roll(self.stackedLocalLinearFns[0][kb],-k,axis=0))[0:(len(self.stackedLocalLinearFns[0][0])-k)]
                        for k in range(1,len(self.stackedLocalLinearFns[0][0])) \
                    ] \
                )
        
        self.checkerGroup = Group(NodeCheckerGenericReach.NodeCheckerGenericReach)

        stat = self.checkerGroup.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, self.selectorMats, awaitable=True)
        stat.get()

        self.poset = Chare(posetFastCharm.Poset,args=[self.checkerGroup,[],False,None,50],onPE=charm.myPe())
        
        stat = self.poset.initialize(self.localLinearFns, self.pt, self.inputConstraintsA, self.inputConstraintsb, awaitable=True)
        stat.get()

        self.copyTime = 0
        self.posetTime = 0
        self.workerInitTime = 0















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