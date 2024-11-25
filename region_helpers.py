import numpy as np
import encapsulateLP
from copy import copy, deepcopy
import posetFastCharm_numba
import vectorSet.vectorSet as vectorSet
from itertools import chain

class flipConstraints:

    def __init__(self, nA, nb, pt, fA=None, fb=None, tol=1e-9,rTol=1e-9, normalize=None):
        if not normalize is None and not ( type(normalize) is float and normalize > 0.0 ):
            raise ValueError('normalize option must be None or a float > 0.0')
        v = nA @ pt
        v = v.flatten() - nb.flatten()
        self.flipMapN = np.where(v<tol,-1,1)
        self.flipMapSetNP = np.nonzero(self.flipMapN < 0)[0]
        self.flipMapSet = frozenset(self.flipMapSetNP)
        self.nA = np.diag(self.flipMapN) @ nA
        self.nb = np.diag(self.flipMapN) @ nb
        if np.any( np.abs(v) <= tol ):
            updatePoint = True
        else:
            updatePoint = False
        self.N = len(nA)
        self.allN = self.N
        self.d = len(nA[0])
        self.pt = pt
        self.tol = tol
        self.rTol = rTol
        self.normalize = normalize

        if (fA is not None) and (fb is not None):
            v = fA @ pt
            v = v.flatten() - fb.flatten()
            if len(np.flatnonzero(v<-tol)) > 0:
                raise ValueError('Supplied point must satisfy all specified \'fixed\' constraints -- i.e. fA @ pt >= fb !')
            self.fA = fA
            self.fb = fb
            # create a vector set for the fixed constraints
            self.fSet = vectorSet.vectorSet( np.hstack((-1*self.fb,self.fA)) )
            self.allConstraints = np.vstack( ( np.hstack((-1*self.nb,self.nA)), self.fSet.getUniqueRows() ) )

        else:
            self.fA = None
            self.fb = None
            self.allConstraints = np.hstack((-self.nb,self.nA))
            self.fSet = None

        self.allNrms = np.ones((self.allConstraints.shape[0],1),dtype=np.float64)
        if not normalize is None:
            self.allNrms = self.normalize / np.linalg.norm(self.allConstraints[:,1:],axis=1).reshape(-1,1)
            self.allConstraints = self.allNrms * self.allConstraints
        else:
            self.normalize = None
        self.nrms = self.allNrms

        # Create a vector set for the main hyperplanes
        self.hyperSet = vectorSet.vectorSet(self.allConstraints[:self.N,:])

        # self.root = bytearray( np.packbits(np.full(self.N,0,dtype=bool),bitorder='little') )
        # self.root = int.from_bytes(self.root, 'little')
        self.root = tuple()
        self.constraints = self.allConstraints
        self.nonRedundantHyperplanes = np.array(self.hyperSet.uniqueRowIdx,dtype=np.int64) if self.fSet is None \
                                                            else np.array(self.hyperSet.subtractSet(self.fSet),dtype=np.int64)
        self.N = len(self.nonRedundantHyperplanes)
        if self.N < self.allN:
            print(f'WARNING: some hyperplanes are duplicates or correspond to fixed boundary hyperplanes!')
        self.constraints = np.vstack([ self.constraints[self.nonRedundantHyperplanes,:], self.constraints[self.allN:,:]])
        if updatePoint:
            self.pt = findInteriorPoint( self.constraints, tol=self.tol, rTol=self.rTol )
            print(f'\nWARNING: Perturbing provided initial point to:\n{self.pt}\n')
            assert not self.pt is None, f'Unable to update anchor point to interior of root region!'
        self.nrms = np.vstack([ self.nrms[self.nonRedundantHyperplanes,], self.nrms[self.allN:,] ])
        self.redundantFlips = np.full(self.allN,-1,dtype=np.int64)
        self.redundantFlips[self.nonRedundantHyperplanes,] = np.ones_like(self.nonRedundantHyperplanes,dtype=np.int64)
        self.wholeBytes = self.N // 8
        self.tailBits = self.N % 8
        self.wholeBytesAllN = self.wholeBytes
        self.tailBitsAllN = self.tailBits
        self.rebasePt = None
        self.baseN = None

    def serialize(self):
        self.hyperSet.serialize()
        if not self.fSet is None:
            self.fSet.serialize()
        return self
    def deserialize(self):
        self.hyperSet.deserialize()
        if not self.fSet is None:
            self.fSet.deserialize()
        return self

    def copy(self):
        return deepcopy(self)

    def insertHyperplane(self,newA,newb):
        newSign = None
        newHyperplane = np.hstack([-newb, newA])
        newSign = 1 if newA.reshape(1,-1) @ self.pt >= newb - self.tol else -1
        if np.abs(newA.reshape(1,-1) @ self.pt - newb) <= self.tol:
            updatePoint = True
        else:
            updatePoint = False
        self.flipMapN = np.hstack([self.flipMapN, np.array([newSign],dtype=np.int64)])
        self.flipMapSetNP = np.nonzero(self.flipMapN < 0)[0]
        self.flipMapSet = frozenset(self.flipMapSetNP)
        # self.baseN allows us to track which hyperplanes were added using the insertHyperplane method
        if self.baseN is None:
            self.baseN = self.N
        self.nA = np.vstack([self.nA, newSign * newA])
        self.nb = np.vstack([self.nb, newSign * newb])
        if self.fA is None:
            self.allConstraints = np.hstack((-self.nb,self.nA))
        else:
            self.allConstraints = np.vstack( ( np.hstack((-1*self.nb,self.nA)), self.allConstraints[self.allN:,:] ) )

        self.allNrms = np.vstack([ self.allNrms[:self.allN,], np.array([[1]],dtype=np.int64), self.allNrms[self.allN:,] ])
        if not self.normalize is None:
            self.allNrms[self.allN,] = self.normalize / np.linalg.norm(self.allConstraints[self.allN,1:].reshape(1,-1),axis=1).reshape(-1,1)
            self.allConstraints = self.allNrms * self.allConstraints
        else:
            self.normalize = None

        self.allN += 1

        print(f'newHyperplane = {newHyperplane}; {self.fSet.isElem(newHyperplane)}')

        if (not self.fSet is None and not self.fSet.isElem(newHyperplane)) and self.hyperSet.insertRow(newHyperplane):
            self.nonRedundantHyperplanes = np.array(self.nonRedundantHyperplanes.tolist() + [self.allN-1], dtype=np.int64)
            self.N = len(self.nonRedundantHyperplanes)

            self.constraints = np.vstack([ self.constraints[self.nonRedundantHyperplanes,:], self.constraints[self.allN:,:]])
            if updatePoint:
                self.pt = findInteriorPoint( self.constraints, tol=self.tol, rTol=self.rTol )
                print(f'\nWARNING: Perturbing provided initial point to:\n{self.pt} after insertion!\n')
                assert not self.pt is None, f'Unable to update anchor point to interior of root region!'
            self.nrms = np.vstack([ self.nrms[self.nonRedundantHyperplanes,], self.nrms[self.allN:,] ])
            self.redundantFlips = np.full(self.allN,-1,dtype=np.int64)
            self.redundantFlips[self.nonRedundantHyperplanes,] = np.ones_like(self.nonRedundantHyperplanes,dtype=np.int64)

            self.wholeBytes = self.N // 8
            self.tailBits = self.N % 8
            self.wholeBytesAllN = self.allN // 8
            self.tailBitsAllN = self.allN % 8
            return True
        else:
            return False

    def filterParallel(self, vec):
        if not isinstance(vec, np.ndarray) or vec.size != self.d + 1:
            raise ValueError(f'Query vector must be a numpy array with {self.d} elements!')
        vec = vec.copy().flatten()
        pars = self.hyperSet.listParallel(vec)
        retIdenHypers = []
        retParHypers = []
        retVal = np.ones((self.N,),dtype=np.bool_)
        if pars is not None:
            idenHypers = list(set(pars[0]) & self.hyperSet.uniqRowIdxSet.keys())
            parHypers = {self.hyperSet.uniqRowIdxSet[i] for i in (set(pars[1]) & self.hyperSet.uniqRowIdxSet.keys())}
            for idx, i in enumerate(self.nonRedundantHyperplanes):
                if len(idenHypers) > 0 and i == self.hyperSet.uniqRowIdxSet[idenHypers[0]]:
                    retIdenHypers.append(idx)
                    retVal[idx] = False
                    # There should only be on instance of any given hyperplane represented in
                    # self.nonRedundantHyperplanes, since we're using a vectorSet there
                if i in parHypers:
                    retParHypers.append(idx)
                    retVal[idx] = False
        retIdenFHypers = []
        retParFHypers = []
        retFVal = np.ones((self.constraints.shape[0]-self.N,),dtype=np.bool_)
        if self.fSet is not None:
            pars = self.fSet.listParallel(vec)
            if pars is not None:
                idenFHypers = list(set(pars[0]) & self.fSet.uniqRowIdxSet.keys())
                if len(idenFHypers) > 0:
                    idx = self.fSet.uniqRowIdxSet[idenFHypers[0]]
                    retIdenFHypers.append(idx + self.N)
                    retFVal[idx] = False
                    # There should only be on instance of any given hyperplane represented in
                    # self.nonRedundantHyperplanes, since we're using a vectorSet there
                parFHypers = set(pars[1]) & self.fSet.uniqRowIdxSet.keys()
                for idx in [self.fSet.uniqRowIdxSet[i] for i in parFHypers]:
                    retParFHypers.append(idx + self.N)
                    retFVal[idx] = False
        return retIdenHypers + retIdenFHypers, retParHypers + retParFHypers, list(np.nonzero(retVal)[0]) + list(self.N + np.nonzero(retFVal)[0])

    def projectConstraints(self,hyperIn,subIdx=None,regSpec=None,extraConstr=None,allN=False):
        if not isinstance(hyperIn, np.ndarray) or hyperIn.size != self.d + 1:
            raise ValueError(f'Projection hyperplane vector must be a numpy array with {self.d} elements!')
        hyper = hyperIn.copy().flatten()

        idenHypers, parHypers, diffHypers = self.filterParallel(hyper)
        origParHypers = parHypers

        if regSpec is None:
            H = self.constraints.copy()
        else:
            if isinstance(regSpec,bytearray):
                nodeBytes = tuple(self.bytesToList(regSpec,allN=allN))
            else:
                nodeBytes = regSpec
            if allN:
                nodeBytes = self.collapseRegion( regSpec )
            H = self.constraints.copy()
            H[nodeBytes,:] = -H[nodeBytes,:]

        # Test each parallel hyperplane to see if it is compatible with (on the correct side of)
        # the projection constraint
        testPt = -(hyperIn[0] / np.linalg.norm(hyperIn[1:])**2) * hyperIn[1:].reshape(-1,1)
        unsatPars = np.any( ( -H[parHypers,1:] @ testPt > H[parHypers,0].reshape(-1,1) + self.tol ).flatten() )
        H = H[diffHypers,:]

        extraDiffHypers = None
        unsatParsExtra = []
        if extraConstr is not None:
            extraSet = vectorSet.vectorSet( extraConstr )
            extraSetRows = extraSet.getUniqueRows()
            pars = extraSet.listParallel(hyper)
            retIdenHypers = []
            retParHypers = []
            retVal = np.ones((extraSet.Nunique,),dtype=np.bool_)
            if pars is not None:
                retParHypers = list(pars[0])
                retIdenHypers = list(pars[1])
                eIdenHypers = set(pars[0])
                eParHypers = set(pars[1])
                extraDiffHypers = sorted(list( (set(range(extraSet.Nunique))-eIdenHypers)-eParHypers ))
                # Test each parallel hyperplane to see if it is compatible with (on the correct side of)
                # the projection constraint
                unsatParsExtra = np.any( ( -extraSetRows[retParHypers,1:] @ testPt > extraSetRows[retParHypers,0].reshape(-1,1) + self.tol ).flatten() )
                if len(extraDiffHypers) > 0 and unsatParsExtra:
                    H = np.vstack([H,extraConstr[extraDiffHypers,:].reshape((len(extraDiffHypers),extraConstr.shape[1]))])
            else:
                H = np.vstack([H,extraConstr])

        return ( projectConstraints(H, hyper, subIdx=subIdx) + (diffHypers, extraDiffHypers) ) if not unsatPars and not unsatParsExtra else None

    # This method returns flips of the hyperplanes *as provided* to obtain the region
    # specified by nodeBytes.
    # This is useful when you don't just care about the region specification but the
    # hyperplanes themselves, as in e.g. FastBATLLNN (there the sign of the hyperplanes
    # means something for the output of the TLL).
    def translateRegion(self,nodeBytesInt, allN=True):
        if isinstance(nodeBytesInt,bytearray):
            nodeBytes = tuple(self.bytesToList(nodeBytesInt))
        else:
            nodeBytes = nodeBytesInt
        regSet = np.full(self.allN, True, dtype=bool)
        regSet[tuple(self.flipMapSet),] = np.full(len(self.flipMapSet),False,dtype=bool)
        # sel = self.nonRedundantHyperplanes[nodeBytes,]
        sel = np.array(sorted(list(chain.from_iterable( \
                   [ self.hyperSet.expandDuplicates(h) for h in self.nonRedundantHyperplanes[nodeBytes,] ] \
              ))),dtype=np.int64)
        regSet[sel,] = np.full(len(sel),False,dtype=bool)
        unflipped = posetFastCharm_numba.is_in_set(self.flipMapSetNP,sel.tolist())
        regSet[unflipped,] = np.full(len(unflipped),True,dtype=bool)
        if not allN:
            regSet = regSet[self.nonRedundantHyperplanes,]
        return np.nonzero(regSet)[0]

    # This method accepts region specifications *relative* to the original flipping
    # (i.e. flipped so that self.pt is on the positive side of all hyperplanes).
    # These region specifications are the ones output by the poset enumerator for example.
    def regionInteriorPoint(self,nodeBytes, allN=False):
        return findInteriorPoint(self.getRegionConstraints(nodeBytes, allN=False))

    # This method also accepts region specifications relative to the original flipping
    # ** It will produce INCORRECT results if fed with the output of translateRegion!! **
    def getRegionConstraints(self, nodeBytesInt, allN=True):
        if isinstance(nodeBytesInt,bytearray):
            nodeBytes = tuple(self.bytesToList(nodeBytesInt))
        else:
            nodeBytes = nodeBytesInt
        H = self.allConstraints.copy() if allN else self.constraints.copy()
        regSet = self.insertRedundant(nodeBytes) if allN and self.N != self.allN else nodeBytes
        H[regSet,:] = -H[regSet,:]
        return H

    # This helper method is only meant to be called in getRegionConstraints above
    def insertRedundant(self, nodeBytesInt):
        print(nodeBytesInt)
        if isinstance(nodeBytesInt,bytearray):
            nodeBytes = tuple(self.bytesToList(nodeBytesInt))
        else:
            nodeBytes = nodeBytesInt
        sel = sorted(list(chain.from_iterable( \
                   [ self.hyperSet.expandDuplicates(h) for h in self.nonRedundantHyperplanes[nodeBytes,] ] \
              )))
        return tuple(sel) #tuple(self.nonRedundantHyperplanes[nodeBytes,])

    def setRebase(self, rebasePoint):
        self.rebasePt = rebasePoint
        v = self.constraints[:self.N, 1:] @ self.rebasePt + self.constraints[:self.N, 0].reshape(-1,1)
        self.rebaseSet = frozenset(np.nonzero(v.flatten() < -self.tol)[0])
        v = self.allConstraints[:self.allN, 1:] @ self.rebasePt + self.allConstraints[:self.allN, 0].reshape(-1,1)
        self.rebaseSetAllN = frozenset(np.nonzero(v.flatten() < -self.tol)[0])

    def rebaseRegion(self, nodeBytesInt, allN=False):
        if self.rebasePt is None:
            return None
        if isinstance(nodeBytesInt,bytearray):
            nodeBytes = tuple(self.bytesToList(nodeBytesInt))
        else:
            nodeBytes = nodeBytesInt
        regSet = set(nodeBytes)
        rebaseSet = self.rebaseSetAllN if allN else self.rebaseSet
        doubleFlip = rebaseSet & regSet
        retTup = tuple(sorted(list( (rebaseSet - doubleFlip) | (regSet - doubleFlip) )))
        return tupToBytes(retTup, self.wholeBytesAllN if allN else self.wholeBytes, self.tailBitsAllN if allN else self.tailBits), retTup

    # def expandRegion(self, nodeBytesInt):
    #     if isinstance(nodeBytesInt,bytearray):
    #         nodeBytes = tuple(self.bytesToList(nodeBytesInt))
    #     else:
    #         nodeBytes = nodeBytesInt
    #     return nodeBytes

    # def collapseRegion(self, nodeBytesInt):
    #     if isinstance(nodeBytesInt,bytearray):
    #         nodeBytes = tuple(self.bytesToList(nodeBytesInt))
    #     else:
    #         nodeBytes = nodeBytesInt
    #     return nodeBytes
    def expandRegion(self,nodeBytesInt):
        if isinstance(nodeBytesInt,bytearray):
            nodeBytes = tuple(self.bytesToList(nodeBytesInt))
        else:
            nodeBytes = nodeBytesInt
        regSet = np.full(self.allN,False,dtype=bool)
        sel = sorted(list(chain.from_iterable( \
                   [ self.hyperSet.expandDuplicates(h) for h in self.nonRedundantHyperplanes[nodeBytes,] ] \
              )))
        for ii in sel:
            regSet[sel] = True
        return tuple(np.nonzero(regSet)[0])

    def collapseRegion(self,nodeBytesInt):
        if isinstance(nodeBytesInt,bytearray):
            nodeBytes = tuple(self.bytesToList(nodeBytesInt))
        else:
            nodeBytes = nodeBytesInt
        regSet = np.full(self.allN,False, dtype=bool)
        regSet[nodeBytes,] = np.full(len(nodeBytes),True,dtype=bool)
        return tuple(np.nonzero(regSet[self.nonRedundantHyperplanes,])[0])

    def bytesToList(self, nodeBytes, allN=False):
        return bytesToList(nodeBytes, self.wholeBytes if not allN else self.wholeBytesAllN, self.tailBits if not allN else self.tailBitsAllN)
    def tupToBytes(self, nodeBytes, allN=False):
        return tupToBytes(nodeBytes, self.wholeBytes if not allN else self.wholeBytesAllN, self.tailBits if not allN else self.tailBitsAllN)

    def computeRelativeRegion(self, nodeBytes, flips, allN=False):
        # Warning: will not check for multiple flips of the same hyperplane!
        N = self.N if not allN else self.allN
        for fl in flips:
            if fl >= N:
                raise ValueError(f'Index of hyperplane to flip must be less than {N} (invoked with allN = {self.allN})')
        if type(nodeBytes) == bytearray or type(nodeBytes) == bytes:
            boolIdxNoFlip = bytearray(copy(nodeBytes))
            for fl in flips:
                boolIdxNoFlip[fl//8] = boolIdxNoFlip[fl//8] ^ 1<<(fl % 8)
            return type(nodeBytes)(boolIdxNoFlip)
        else:
            INTrep = set(nodeBytes)
            for fl in flips:
                if fl in INTrep:
                    INTrep.remove(fl)
                else:
                    INTrep.add(fl)
            if type(nodeBytes) == np.ndarray:
                return np.array(sorted(list(nodeBytes)),dtype=nodeBytes.dtype)
            else:
                return type(nodeBytes)(INTrep)


class flipConstraintsReduced(flipConstraints):

    def __init__(self, nA, nb, pt, fA=None, fb=None, tol=1e-9,rTol=1e-9,normalize=None):
        print(f'WARNING: The class flipConstraintsReduced is deprecated, and may not work correctly!')
        super().__init__(nA, nb, pt, fA=fA, fb=fb, tol=tol, rTol=rTol, normalize=normalize)
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
        self.wholeBytes = self.N // 8
        self.tailBits = self.N % 8


class flipConstraintsReducedMin(flipConstraints):

    def __init__(self, nA, nb, pt, fA=None, fb=None, tol=1e-9,rTol=1e-9, normalize=None):
        super().__init__(nA, nb, pt, fA=fA, fb=fb, tol=tol, rTol=rTol, normalize=normalize)
        if self.fA is None:
            return

        mat = copy(self.constraints[(self.N-1):,:])
        # self.redundantFlips = np.full(self.allN,-1,dtype=np.float64)
        for k in range(self.N):
            mat[0,:] = self.constraints[k,:] # same as self.allConstraints[self.nonRedundantHyperplanes[k],:]
            if len(lpMinHRep(mat,None,[0])) == 0:
                self.redundantFlips[self.nonRedundantHyperplanes[k]] = -1
            else:
                self.redundantFlips[self.nonRedundantHyperplanes[k]] = 1
        self.nonRedundantHyperplanes = np.nonzero(self.redundantFlips > 0)[0]

        self.nrms = np.vstack([ self.allNrms[self.nonRedundantHyperplanes,].reshape(-1,1), self.allNrms[self.allN:,].reshape(-1,1) ])
        self.constraints = np.vstack( \
                                    ( \
                                        self.allConstraints[self.nonRedundantHyperplanes,:], \
                                        self.allConstraints[self.allN:,:] \
                                    ) \
                                )
        self.N = len(self.nonRedundantHyperplanes)
        self.wholeBytes = self.N // 8
        self.tailBits = self.N % 8
        self.wholeBytesAllN = self.allN // 8
        self.tailBitsAllN = self.allN % 8

        self.root = tuple()

    def insertHyperplane(self,newA,newb):
        N = self.N
        if self.baseN is None:
            self.baseN = self.N
        if super().insertHyperplane(newA,newb):
            mat = copy(self.constraints[(self.N-1):,:])
            # self.redundantFlips = np.hstack([self.redundantFlips, np.array([-1],dtype=np.int64)])
            for k in [self.N-1]:
                mat[0,:] = self.constraints[k,:]
                if len(lpMinHRep(mat,None,[0])) == 0:
                    self.redundantFlips[self.nonRedundantHyperplanes[k]] = -1
                else:
                    self.redundantFlips[self.nonRedundantHyperplanes[k]] = 1
            self.nonRedundantHyperplanes = np.nonzero(self.redundantFlips > 0)[0]

            self.nrms = np.vstack([ self.allNrms[self.nonRedundantHyperplanes,].reshape(-1,1), self.allNrms[self.allN:,].reshape(-1,1) ])
            self.constraints = np.vstack( \
                                        ( \
                                            self.allConstraints[self.nonRedundantHyperplanes,:], \
                                            self.allConstraints[self.allN:,:] \
                                        ) \
                                    )
            self.N = len(self.nonRedundantHyperplanes)
            self.wholeBytes = self.N // 8
            self.tailBits = self.N % 8
            self.wholeBytesAllN = self.allN // 8
            self.tailBitsAllN = self.allN % 8
            return True
        else:
            return False

    # def translateRegion(self,nodeBytesInt, allN=True):
    #     if isinstance(nodeBytesInt,bytearray):
    #         nodeBytes = tuple(self.bytesToList(nodeBytesInt))
    #     else:
    #         nodeBytes = nodeBytesInt
    #     regSet = np.full(self.allN, True, dtype=bool)
    #     regSet[tuple(self.flipMapSet),] = np.full(len(self.flipMapSet),False,dtype=bool)
    #     # sel = self.nonRedundantHyperplanes[nodeBytes,]
    #     sel = np.array(sorted(list(chain.from_iterable( \
    #                [ self.hyperSet.expandDuplicates(h) for h in self.nonRedundantHyperplanes[nodeBytes,] ] \
    #           ))),dtype=np.int64)
    #     regSet[sel,] = np.full(len(sel),False,dtype=bool)
    #     unflipped = posetFastCharm_numba.is_in_set(self.flipMapSetNP,sel.tolist())
    #     regSet[unflipped,] = np.full(len(unflipped),True,dtype=bool)
    #     if not allN:
    #         regSet = regSet[self.nonRedundantHyperplanes,]
    #     return np.nonzero(regSet)[0]

    # def expandRegion(self,nodeBytesInt):
    #     if isinstance(nodeBytesInt,bytearray):
    #         nodeBytes = tuple(self.bytesToList(nodeBytesInt))
    #     else:
    #         nodeBytes = nodeBytesInt
    #     regSet = np.full(self.allN,False,dtype=bool)
    #     sel = sorted(list(chain.from_iterable( \
    #                [ self.hyperSet.expandDuplicates(h) for h in self.nonRedundantHyperplanes[nodeBytes,] ] \
    #           )))
    #     for ii in sel:
    #         regSet[sel] = True
    #     return tuple(np.nonzero(regSet)[0])

    # def collapseRegion(self,nodeBytesInt):
    #     if isinstance(nodeBytesInt,bytearray):
    #         nodeBytes = tuple(self.bytesToList(nodeBytesInt))
    #     else:
    #         nodeBytes = nodeBytesInt
    #     regSet = np.full(self.allN,False, dtype=bool)
    #     regSet[nodeBytes,] = np.full(len(nodeBytes),True,dtype=bool)
    #     return tuple(np.nonzero(regSet[self.nonRedundantHyperplanes,])[0])

def byteLenFromN(N):
    return  N // 8  ,   N % 8

def recodeRegNewN(strip, reg, N):
    if isinstance(reg,tuple) or isinstance(reg,list):
        reg = tupToBytes( reg, *byteLenFromN(N) )
    if strip == 0:
        return reg, tuple(bytesToList(reg,*byteLenFromN(N))), N
    elif strip < 0:
        newN = N + strip
        assert newN > 0, f'Can\'t remove {strip} hyperplane from a list of {N}'
        newWholeBytes, newTailBits = byteLenFromN(newN)
        newReg = reg[:(newWholeBytes + (1 if newTailBits > 0 else 0))]
        if newTailBits > 0:
            newReg[-1] = newReg[-1] & (2**newTailBits - 1)
    elif strip > 0:
        newN = N + strip
        newWholeBytes, newTailBits = byteLenFromN(newN)
        newReg = bytearray(b'\x00') *  (newWholeBytes + (1 if newTailBits != 0 else 0))
        newReg[:len(reg)] = reg[:]
    else:
        print(f'Error...')
        return None, None, None
    return newReg, tuple(bytesToList(newReg,newWholeBytes,newTailBits)), newN

# H2 is a CDD-style matrix specifying inequality constraints, and intIdx is a list of indices of inequalities to check for redundancy
# The return value is a list of indices into the list intIdx specifying which of those constraints are non-redundant
def lpMinHRep(H2,constraint_list_in,intIdx,solver='glpk',safe=False,lpObj=None,tol=1e-9,rTol=1e-9):
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
            or (not safe and (status == 'primal infeasible' or np.all(-H2[intIdx[idx],1:]@x - H2[intIdx[idx],0] <= tol + rTol * np.abs(H[:,0].reshape(-1,1))))):
            # inequality is redundant, so skip it
            constraint_list[offsetIdx] = False
        else:
            to_keep.append(idx)

    return to_keep

# H2 is a CDD-style matrix specifying inequality constraints
# Function returns an interior point to the associated region, if one exists, and None otherwise
def findInteriorPointFull(H2,allowEquality=None,solver='glpk',lpObj=None,tol=1e-7,rTol=1e-7,lpopts={}):
    lpopts = deepcopy(lpopts)
    lpopts['solver'] = solver

    if lpObj is None:
       lpObj = encapsulateLP.encapsulateLP()
       lpObj.initSolver(solver=solver,opts={'dim':(H2.shape[1])}),

    n = H2.shape[1]-1
    N = H2.shape[0]

    H = copy(H2)
    obj = np.zeros(n+1,dtype=np.float64)
    obj[n] = -1
    eqConstraints = np.ones((N,1),dtype=np.float64)
    if not allowEquality is None:
        eqConstraints[allowEquality,] = np.zeros_like(eqConstraints[allowEquality,])
    constraints = np.hstack([\
            np.hstack([H[:,0], 1.0, 0.0]).reshape(-1,1), \
            np.vstack([ \
                np.hstack([-H[:,1:], eqConstraints]), \
                -obj, \
                obj \
            ]) \
        ])
    constraints = constraints / np.linalg.norm(constraints,axis=1).reshape(-1,1)
    obj = constraints[-1,1:]
    status, sol = lpObj.runLP( \
                    obj, \
                    constraints[:,1:], \
                    constraints[:,0], \
                    lpopts = lpopts \
                )
    return status, (np.frombuffer(sol) if status=='optimal' else None)

def findInteriorPoint(H2,allowEqualityConstraints=None,solver='glpk',lpObj=None,tol=1e-7,rTol=1e-7,lpopts={}):
    status, sol = findInteriorPointFull(H2,solver=solver,lpObj=lpObj,allowEquality=allowEqualityConstraints,tol=tol,rTol=rTol,lpopts=lpopts)
    n = H2.shape[1]-1
    if status == 'optimal' and sol[-1] > tol:
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


def regionBBox(H,solver='glpk',lpObj=None,tol=1e-7,rTol=1e-7,regionCheck=True):
    if lpObj is None:
       lpObj = encapsulateLP.encapsulateLP()
       lpObj.initSolver(solver=solver)
    if regionCheck:
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
                    lpopts = {'solver':solver, 'fallback':'glpk'} \
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
    retSamples = np.zeros((n,numSamples),dtype=np.float64)
    maxSamples = 0
    while maxSamples < numSamples:
        diff = box[:,1].reshape(-1,1) - box[:,0].reshape(-1,1)
        summ = box[:,1].reshape(-1,1) + box[:,0].reshape(-1,1)
        samps = diff * np.random.random((n,numSamples)) + 0.5 * summ - 0.5 * diff
        locs = np.nonzero(np.all(-H[:,1:] @ samps - H[:,0].reshape(-1,1) <= 0,axis=0))[0]
        newMaxSamples = min(maxSamples + len(locs),numSamples)
        locs = locs[:(newMaxSamples - maxSamples)]
        retSamples[:,maxSamples:newMaxSamples] = samps[:,locs]
        maxSamples = newMaxSamples
    return retSamples

class DegenerateHyperplane(ValueError): pass

def projectConstraints(H,hyperIn,subIdx=None,tol=1e-8,rTol=1e-8):
    hyper = hyperIn.copy().flatten()
    hyper[1:] = -hyper[1:]
    assert H.shape[1] == hyper.shape[0]
    # assert H.shape[1] > 2, 'Projecting constraints over 1-d results in points'
    tempIdx = np.nonzero(hyper[1:])[0]
    if len(tempIdx) == 0:
        raise DegenerateHyperplane
    if subIdx is None:
        subIdx = tempIdx[0]
        # print(f'Local subIdx {subIdx}')
    else:
        assert subIdx in tempIdx
    retH = np.zeros((H.shape[0],H.shape[1]-1),dtype=np.float64)
    hyperSlice = np.zeros((hyper.shape[0]-2,),dtype=np.float64)
    hyperSlice[0:subIdx] = hyper[1:(subIdx+1)]
    hyperSlice[subIdx:] = hyper[(subIdx+2):]

    A = -H[:,1:]
    b = H[:,0]
    # A @ x <= b
    retH[:,1:(subIdx+1)] = -A[:,0:subIdx] + A[:,subIdx].reshape(-1,1) * (1/hyper[subIdx+1]) * hyperSlice[0:subIdx]
    retH[:,(subIdx+1):] = -A[:,(subIdx+1):] + A[:,subIdx].reshape(-1,1) * (1/hyper[subIdx+1]) * hyperSlice[subIdx:]
    retH[:,0] = b - A[:,subIdx] * (1/hyper[subIdx+1]) * hyper[0]
    return retH, subIdx

def liftPoint(x,hyperInT,subIdx):
    hyperIn = hyperInT.flatten()
    assert subIdx < hyperIn.shape[0] - 1, f'subIdx must be one less than the dimension of the hyperplane'
    assert len(x.shape) == 2, f'Input point(s) must be a column vector or matrix of column vectors'
    assert x.shape[0] == hyperIn.shape[0] - 2, f'Input point(s) must be in 1 dimension less than input hyperplane'
    retVal = np.zeros((hyperIn.shape[0]-1,x.shape[1]),dtype=np.float64)
    retVal[subIdx] = ( \
                            hyperIn[0] + hyperIn[1:(subIdx+1)].reshape(1,-1) @ x[:subIdx,:].reshape(-1,x.shape[1]) + \
                            hyperIn[(subIdx+2):].reshape(1,-1) @ x[subIdx:,:].reshape(-1,x.shape[1]) \
                     )/(-hyperIn[subIdx+1])
    retVal[:subIdx,:] = x[:subIdx,:]
    retVal[(subIdx+1):,:] = x[subIdx:,:]
    return retVal

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

