import numpy as np
import encapsulateLP
from copy import copy
import posetFastCharm_numba


class flipConstraints:

    def __init__(self, nA, nb, pt, fA=None, fb=None, tol=1e-9,rTol=1e-9):
        v = nA @ pt
        v = v.flatten() - nb.flatten()
        self.flipMapN = np.where(v<0,-1,1)
        self.flipMapSetNP = np.nonzero(self.flipMapN < 0)[0]
        self.flipMapSet = frozenset(self.flipMapSetNP)
        self.nA = np.diag(self.flipMapN) @ nA
        self.nb = np.diag(self.flipMapN) @ nb
        self.N = len(nA)
        self.d = len(nA[0])
        self.pt = pt
        self.tol = tol
        self.rTol = rTol

        if (fA is not None) and (fb is not None):
            v = fA @ pt
            v = v.flatten() - fb.flatten()
            if len(np.flatnonzero(v<-tol)) > 0:
                raise ValueError('Supplied point must satisfy all specified \'fixed\' constraints -- i.e. fA @ pt >= fb !')
            self.fA = fA
            self.fb = fb
            self.constraints = np.vstack( ( np.hstack((-1*self.nb,self.nA)), np.hstack((-1*self.fb,self.fA)) ) )

        else:
            self.fA = None
            self.fb = None
            self.constraints = np.hstack((-self.nb,self.nA))

        # self.root = bytearray( np.packbits(np.full(self.N,0,dtype=bool),bitorder='little') )
        # self.root = int.from_bytes(self.root, 'little')
        self.root = tuple()
        self.allConstraints = self.constraints
        self.allN = self.N
        self.redundantFlips = np.full(self.N,1,dtype=np.int64)
        self.nonRedundantHyperplanes = np.arange(self.N)
        self.wholeBytes = (self.N + 7) // 8
        self.tailBits = self.N - 8*(self.N // 8)
        self.rebasePt = None

    # This method returns flips of the hyperplanes *as provided* to obtain the region
    # specified by nodeBytes.
    # This is useful when you don't just care about the region specification but the
    # hyperplanes themselves, as in e.g. FastBATLLNN (there the sign of the hyperplanes
    # means something for the output of the TLL).
    def translateRegion(self,nodeBytes, allN=True):
        regSet = np.full(self.allN, True, dtype=bool)
        regSet[tuple(self.flipMapSet),] = np.full(len(self.flipMapSet),False,dtype=bool)
        regSet[nodeBytes,] = np.full(len(nodeBytes),False,dtype=bool)
        unflipped = posetFastCharm_numba.is_in_set(self.flipMapSetNP,list(nodeBytes))
        regSet[unflipped,] = np.full(len(unflipped),True,dtype=bool)
        return np.nonzero(regSet)[0]

    # This method accepts region specifications *relative* to the original flipping
    # (i.e. flipped so that self.pt is on the positive side of all hyperplanes).
    # These region specifications are the ones output by the poset enumerator for example.
    def regionInteriorPoint(self,nodeBytes, allN=False):
        return findInteriorPoint(self.getRegionConstraints(nodeBytes, allN=False))

    # This method also accepts region specifications relative to the original flipping
    # ** It will produce INCORRECT results if fed with the output of translateRegion!! **
    def getRegionConstraints(self, nodeBytes, allN=True):
        H = self.allConstraints.copy() if allN else self.constraints.copy()
        regSet = self.insertRedundant(nodeBytes) if allN and self.N != self.allN else nodeBytes
        H[regSet,] = -H[regSet,]
        return H

    # This helper method is only meant to be called in getRegionConstraints above
    def insertRedundant(self, nodeBytes):
        return tuple(self.nonRedundantHyperplanes[nodeBytes,])

    def setRebase(self, rebasePoint):
        self.rebasePt = rebasePoint
        v = self.constraints[:self.N, 1:] @ self.rebasePt + self.constraints[:self.N, 0].reshape(-1,1)
        self.rebaseSet = frozenset(np.nonzero(v.flatten() < -self.tol)[0])

    def rebaseRegion(self, nodeBytes):
        if self.rebasePt is None:
            return None
        regSet = set(nodeBytes)
        doubleFlip = self.rebaseSet & regSet
        retTup = tuple(sorted(list( (self.rebaseSet - doubleFlip) | (regSet - doubleFlip) )))
        return tupToBytes(retTup, self.wholeBytes, self.tailBits), retTup


class flipConstraintsReduced(flipConstraints):

    def __init__(self, nA, nb, pt, fA=None, fb=None, tol=1e-9,rTol=1e-9):
        super().__init__(nA, nb, pt, fA=fA, fb=fb, tol=tol, rTol=rTol)
        if self.fA is None:
            return

        mat = copy(self.constraints[(self.N-1):,:])
        self.redundantFlips = np.full(self.N,1,dtype=np.float64)
        for k in range(self.N):
            mat[0,:] = self.constraints[k,:]
            if len(lpMinHRep(mat,None,[0])) == 0:
                self.redundantFlips[k] = -1

        self.nA = np.diag(self.redundantFlips) @ self.nA
        self.nb = np.diag(self.redundantFlips) @ self.nb
        self.constraints = np.vstack( ( np.hstack((-1*self.nb,self.nA)), np.hstack((-1*self.fb,self.fA)) ) )

        self.flipMapN = self.redundantFlips * self.flipMapN
        self.flipMapSetNP = np.nonzero(self.flipMapN < 0)[0]
        self.flipMapSet = frozenset(self.flipMapSetNP)

        self.root = tuple(np.nonzero(self.redundantFlips<0)[0].tolist())
        self.allConstraints = self.constraints
        self.allN = self.N
        self.nonRedundantHyperplanes = np.arange(self.N)
        self.wholeBytes = (self.N + 7) // 8
        self.tailBits = self.N - 8*(self.N // 8)


class flipConstraintsReducedMin(flipConstraints):

    def __init__(self, nA, nb, pt, fA=None, fb=None, tol=1e-9,rTol=1e-9):
        super().__init__(nA, nb, pt, fA=fA, fb=fb, tol=tol, rTol=rTol)
        if self.fA is None:
            return

        mat = copy(self.constraints[(self.N-1):,:])
        self.redundantFlips = np.full(self.N,1,dtype=np.float64)
        for k in range(self.N):
            mat[0,:] = self.constraints[k,:]
            if len(lpMinHRep(mat,None,[0])) == 0:
                self.redundantFlips[k] = -1
        self.nonRedundantHyperplanes = np.nonzero(self.redundantFlips > 0)[0]

        self.allConstraints = copy(self.constraints)
        self.allN = self.N
        self.constraints = np.vstack( ( np.hstack((-1*self.nb[self.nonRedundantHyperplanes,],self.nA[self.nonRedundantHyperplanes,:])), np.hstack((-1*self.fb,self.fA)) ) )
        self.N = len(self.nonRedundantHyperplanes)
        self.wholeBytes = (self.N + 7) // 8
        self.tailBits = self.N - 8*(self.N // 8)

        self.root = tuple()

    def translateRegion(self,nodeBytes, allN=True):
        regSet = np.full(self.allN, True, dtype=bool)
        regSet[tuple(self.flipMapSet),] = np.full(len(self.flipMapSet),False,dtype=bool)
        sel = self.nonRedundantHyperplanes[nodeBytes,]
        regSet[sel,] = np.full(len(sel),False,dtype=bool)
        unflipped = posetFastCharm_numba.is_in_set(self.flipMapSetNP,sel.tolist())
        regSet[unflipped,] = np.full(len(unflipped),True,dtype=bool)
        if not allN:
            regSet = regSet[self.nonRedundantHyperplanes,]
        return np.nonzero(regSet)[0]


# H2 is a CDD-style matrix specifying inequality constraints, and intIdx is a list of indices of inequalities to check for redundancy
# The return value is a list of indices into the list intIdx specifying which of those constraints are non-redundant
def lpMinHRep(H2,constraint_list_in,intIdx,solver='glpk',safe=False,lpObj=None):
    if len(intIdx) == 0:
        return np.full(0,0,dtype=bool)

    if lpObj is None:
       lpObj = encapsulateLP.encapsulateLP()
       lpObj.initSolver(solver=solver)

    restricted = False if constraint_list_in is None else True

    # H2 should be a view into the CDD-formatted H matrix selected by taking boolIdx or intIdx rows thereof
    if safe:
        H = H2 if not restricted else H2[constraint_list_in[0:len(H2)],:]
    else:
        # This version of H has an extra row, that we can use for the another constraint
        H = np.vstack([H2, [H2[0,:]] ]) if not restricted else np.vstack([H2[constraint_list_in,:], [H2[0,:]] ])

    to_keep = []
    constraint_list = np.full(len(H),True,dtype=bool)
    if restricted:
        restIdxs = np.nonzero(constraint_list_in)[0]
        offsetTab = dict(zip(restIdxs,range(len(restIdxs))))
    for idx in range(len(intIdx)):
        if restricted and (not constraint_list_in[intIdx[idx]]):
            continue
        offsetIdx = intIdx[idx] if not restricted else offsetTab[intIdx[idx]]
        # You can test the current hyperplane for consideration by other means at this point
        if not safe:
            # Set the extra row to the negation of the pre-relaxed current constraint
            H[-1,:] = -H2[intIdx[idx],:]
        H[offsetIdx,0] += 1
        status, x = lpObj.runLP( \
                H2[intIdx[idx],1:], \
                -H[constraint_list,1:], \
                H[constraint_list,0], \
                lpopts = {'solver':solver, 'fallback':'glpk'} if solver != 'glpk' else {'solver':'glpk'}, \
                msgID = 'None' \
            )
        H[offsetIdx,0] -= 1

        if status != 'optimal' and (safe or status != 'primal infeasible') and status != 'dual infeasible':
            print('********************  WARNING!!  ********************')
            print('Infeasible or numerical ill-conditioning detected at node -- RESULTS MAY NOT BE ACCURATE!!')
            return [set([]), 0]
        if (safe and -H2[intIdx[idx],1:]@x < H2[intIdx[idx],0]) \
            or (not safe and (status == 'primal infeasible' or np.all(-H2[intIdx[idx],1:]@x - H2[intIdx[idx],0] <= 1e-10))):
            # inequality is redundant, so skip it
            constraint_list[offsetIdx] = False
        else:
            to_keep.append(idx)

    return to_keep

# H2 is a CDD-style matrix specifying inequality constraints
# Function returns an interior point to the associated region, if one exists, and None otherwise
def findInteriorPointFull(H2,solver='glpk',lpObj=None,tol=1e-7,rTol=0):
    if lpObj is None:
       lpObj = encapsulateLP.encapsulateLP()
       lpObj.initSolver(solver=solver,opts={'dim':(H2.shape[1])}),

    n = H2.shape[1]-1
    N = H2.shape[0]

    H = copy(H2)
    obj = np.zeros(n+1,dtype=np.float64)
    obj[n] = -1
    status, sol = lpObj.runLP( \
                    obj, \
                    np.vstack([ \
                               np.hstack([-H[:,1:], np.ones((N,1),dtype=np.float64)]), \
                               -obj, \
                               obj \
                            ]), \
                    np.hstack([H[:,0], 1.0, 0.0]), \
                    lpopts = {'solver':solver}
                )
    return status, (np.frombuffer(sol) if status=='optimal' else None)

def findInteriorPoint(H2,solver='glpk',lpObj=None,tol=1e-7,rTol=0):
    status, sol = findInteriorPointFull(H2,solver=solver,lpObj=lpObj,tol=tol,rTol=rTol)
    n = H2.shape[1]-1
    if status == 'optimal' and sol[-1] > 1e-10:
        sol = np.array(sol)[:n].reshape(-1,1)
        return sol
    else:
        return None

def findInteriorPointOld(H2,solver='glpk',lpObj=None,tol=1e-7,rTol=0):
    if lpObj is None:
       lpObj = encapsulateLP.encapsulateLP()
       lpObj.initSolver(solver=solver)

    n = H2.shape[1]-1
    N = H2.shape[0]

    H = copy(H2)

    status, sol = lpObj.runLP( \
                    np.ones(n,dtype=np.float64), \
                    -H[:,1:], \
                    H[:,0], \
                    lpopts = {'solver':solver}
                )
    if status == 'optimal':

        actHypers = np.nonzero(np.isclose( -H[:,1:] @ sol , H[:,0], atol=tol, rtol=rTol))[0]
        if len(actHypers) == 0:
            return None

        origSol = np.array(sol,dtype=np.float64)
        # print(solList)
        for k in actHypers:
            # Try to get away from the kth active hyperplane
            newStatus, newSol = lpObj.runLP( \
                        -H[k,1:], \
                        -H[:,1:], \
                        H[:,0], \
                        lpopts = {'solver':'glpk'}
                    )
            if np.isclose(H[k,1:]@newSol, H[k,1:]@origSol, atol=tol, rtol=rTol):
                # We were unable to get off hyperplane k, so there is no interior point
                return None
            else:
                # Perturb kth hyperplane into interior of the region (the result is still non-empty,
                # since the set is convex, so origSol and newSol are connected by a segment in the region).
                H[k,0] = 0.5*(H[k,0] - H[k,1:] @ newSol)
        # If we were able to get away from all the active hyperplanes, then the last
        # newSol can be connected to origSol with a segment whose midpoint is in the interior
        # of the region.
        return 0.5*(newSol + origSol)

    else:
        return None


def regionBBox(H,solver='glpk',lpObj=None,tol=1e-7,rTol=1e-7):
    if lpObj is None:
       lpObj = encapsulateLP.encapsulateLP()
       lpObj.initSolver(solver=solver)
    assert not findInteriorPoint(H,solver=solver,lpObj=lpObj,tol=tol,rTol=rTol) is None, 'H matrix specifies an empty region'
    n = H.shape[1]-1
    box = np.array([ [-np.inf, np.inf] for _ in range(n) ],dtype=np.float64)
    objective = np.zeros(n,dtype=np.float64)
    for idx in range(n):
        for direc, idx2 in [(1,0),(-1,1)]:
            objective[idx] = direc
            status, sol = lpObj.runLP( \
                    objective, \
                    -H[:,1:], \
                    H[:,0], \
                    lpopts = {'solver':solver} \
                )
            objective[idx] = 0
            box[idx,idx2] = np.array(sol).flatten()[idx]
    return box

def sampleRegion(H,solver='glpk',lpObj=None,tol=1e-7,rTol=1e-7,numSamples=10000):
    if lpObj is None:
       lpObj = encapsulateLP.encapsulateLP()
       lpObj.initSolver(solver=solver)
    box = regionBBox(H,solver=solver,lpObj=lpObj,tol=tol,rTol=rTol)
    # print(box)
    n = box.shape[0]
    diff = box[:,1].reshape(-1,1) - box[:,0].reshape(-1,1)
    summ = box[:,1].reshape(-1,1) + box[:,0].reshape(-1,1)
    samps = diff * np.random.random((n,numSamples)) + 0.5 * summ - 0.5 * diff
    locs = np.nonzero(np.all(-H[:,1:] @ samps - H[:,0].reshape(-1,1) <= 0,axis=0))[0]
    return samps[:,locs].copy()

def projectConstraints(H,hyper,subIdx=None,tol=1e-8,rTol=1e-8):
    hyper = hyper.flatten()
    assert H.shape[1] == hyper.shape[0]
    assert H.shape[1] > 2, 'Projecting constraints over 1-d results in points'
    tempIdx = np.nonzero(hyper[1:])[0]
    assert len(tempIdx) > 0
    if subIdx is None:
        subIdx = tempIdx[0]
        print(f'Local subIdx {subIdx}')
    else:
        assert subIdx in tempIdx
    retH = np.zeros((H.shape[0],H.shape[1]-1),dtype=np.float64)
    hyperSlice = np.zeros((hyper.shape[0]-2,),dtype=np.float64)
    hyperSlice[0:subIdx] = hyper[1:(subIdx+1)]
    hyperSlice[subIdx:] = hyper[(subIdx+2):]

    A = -H[:,1:]
    b = H[:,0]
    # A @ x <= b
    retH[:,1:(subIdx+1)] = -A[:,0:subIdx] + A[:,subIdx].reshape(-1,1) * (1/hyper[subIdx+1]) * hyperSlice
    retH[:,(subIdx+1):] = -A[:,(subIdx+1):] + A[:,subIdx].reshape(-1,1) * (1/hyper[subIdx+1]) * hyperSlice
    retH[:,0] = b - A[:,subIdx] * (1/hyper[subIdx+1]) * hyper[0]
    return retH, subIdx

# Helper functions:

# https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
def hashNodeBytes(nodeBytes):
    chunks = np.array( \
        [int.from_bytes(nodeBytes[idx:min(idx+8,len(nodeBytes))],'little') for idx in range(0,len(nodeBytes),8)], \
        dtype=np.uint64 \
        )
    p = 6148914691236517205 * np.bitwise_xor(chunks, np.right_shift(chunks,32))
    hashInt = 17316035218449499591 * np.bitwise_xor(p, np.right_shift(p,32))
    return int(np.bitwise_xor.reduce(hashInt))

def tupToBytes(INTrep, wholeBytes, tailBits):
    boolIdxNoFlip = bytearray(b'\x00') *  (wholeBytes + (1 if tailBits != 0 else 0))

    for unflipIdx in range(len(INTrep)-1,-1,-1):
        boolIdxNoFlip[INTrep[unflipIdx]//8] = boolIdxNoFlip[INTrep[unflipIdx]//8] | (1<<(INTrep[unflipIdx] % 8))

    return boolIdxNoFlip

def bytesToList(boolIdxNoFlip,wholeBytes,tailBits):
    INTrep = []
    for bIdx in range(wholeBytes + (1 if tailBits != 0 else 0)):
        for bitIdx in range(8 if bIdx < wholeBytes else tailBits):
            if boolIdxNoFlip[bIdx] & ( 1 << bitIdx):
                INTrep.append(8*bIdx + bitIdx)
    return INTrep

