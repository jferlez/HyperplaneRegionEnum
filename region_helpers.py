import numpy as np
import encapsulateLP
from copy import copy


class flipConstraints:

    def __init__(self, nA, nb, pt, fA=None, fb=None):
        v = nA @ pt
        v = v.flatten() - nb.flatten()
        self.flipMapN = np.where(v<0,-1,1)
        self.flipMapSetNP = np.nonzero(self.flipMapN < 0)[0]
        self.flipMapSet = frozenset(self.flipMapSetNP)
        self.nA = np.diag(self.flipMapN) @ nA
        self.nb = np.diag(self.flipMapN) @ nb
        self.N = len(nA)
        self.d = len(nA[0])

        if (fA is not None) and (fb is not None):
            v = fA @ pt
            v = v.flatten() - fb.flatten()
            if len(np.flatnonzero(v<0)) > 0:
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


class flipConstraintsReduced(flipConstraints):

    def __init__(self, nA, nb, pt, fA=None, fb=None):
        super().__init__(nA, nb, pt, fA=fA, fb=fb)
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


class flipConstraintsReducedMin(flipConstraints):

    def __init__(self, nA, nb, pt, fA=None, fb=None):
        super().__init__(nA, nb, pt, fA=fA, fb=fb)
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
        

        self.root = tuple()


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
def findInteriorPoint(H2,solver='glpk',lpObj=None,tol=1e-7,rTol=0):
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






