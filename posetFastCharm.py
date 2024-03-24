import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import cdd
import numpy as np
from copy import copy, deepcopy
import time
import itertools
from functools import partial
from collections import defaultdict
import encapsulateLP
import DistributedHash
import cvxopt
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import sys
import warnings
import numba as nb
import random
# import TLLHypercubeReach
import posetFastCharm_numba
import region_helpers
from region_helpers import hashNodeBytes, tupToBytes, bytesToList

warnings.simplefilter(action = "ignore", category = RuntimeWarning)


class PosetNode(DistributedHash.Node):
    # DO NOT OVERRIDE PARENT'S __init__() method
    # DistributedHash will create a local property with a proxy, self.localProxy, that we can call
    #   (we will use this to make sure that any necessary variables are copied to the required PEs)
    # DistributedHash will also add a property called parentChare to allow acces to data on the hash worker
    def dummy(self):
        pass
    # These methods are optional, and will be called at an appropriate time by DistributedHash if present
    def init(self):
        self.constraints = self.localProxy[self.storePe].getConstraintsOnly(ret=True).get()

    # def update(self):
    #     pass

    # def check(self):
    #     pass
    def update(self, lsb,msb,nodeBytes,N, originPe, face, witness, adj, *args):
        self.face |= set(face)
        if adj and isinstance(adj,dict):
            for ky in adj.keys():
                self.adj[ky] = adj[ky]

    # Do not override these methods:
    def checkForInsert(self):
        return True

    def updateForInsert(self, lsb, msb, nodeBytes, N, originPe, face, witness, adj, *args):
        self.face |= set(face)
        origFace = copy(self.face)
        if not adj is None and isinstance(adj,dict):
            for ky in adj.keys():
                if adj[ky] is None:
                    if ky in self.adj:
                        del self.adj[ky]
                else:
                    self.adj[ky] = adj[ky]
            # Use incomming adj to update face information
            # (But ONLY IF it contains at least one key that is not -1; that structure is used for
            # newly seeded nodes, where we DON'T wish to overwrite face info -- in that case the face
            # corresponds to the incomming faces.)
            # NB: sending an empty adj={} will not alter faces. Faces must be deleted by sending
            # adj[face]=None
            if len(self.adj) >= 1 and tuple(self.adj.keys()) != (-1,):
                self.face = set([ky for ky in self.adj if ky != -1])
        # print(f'    :-:-:-:-:    {charm.myPe()} {nodeBytes}: Updating original face information {origFace} to {self.face} {self.adj}')
        # self.update(lsb, msb, nodeBytes, N, originPe, face, witness, adj, *args)
    def adjFaceCreate(self):
        if not isinstance(self.adj,dict) or len(self.adj) == 0:
            self.adj = {f:self.N for f in self.face}
            self.adj[-1] = self.N


class localVar(Chare):
    def init(self,succGroupProxy,posetPElist):
        self.posetSuccGroupProxy = succGroupProxy
        self.posetPElist = posetPElist
        self.schedCount = 0
        # self.closedCalls = []
        self.skip = False
        self.counterExample = None
    def setConstraintsOnly(self,constraints):
        self.constraints = constraints.deserialize()
        self.schedCount = 0
        self.skip = False
    def getConstraintsOnly(self):
        return self.constraints.serialize()
    # This method **must** be implemented for DistributedHash to work:
    @coro
    def getSchedCount(self):
        return self.schedCount
    def setSkip(self,val):
        # print('Executing setSkip on PE ' + str(charm.myPe()))
        self.skip = val
        # return 37
    @coro
    def reset(self):
        self.skip = False
        self.schedCount = 0
    @coro
    def schedRandomPosetPe(self):
        # self.schedCount += 1
        return random.choice(self.posetPElist)
    @coro
    def checkNode(self, *args):
        return True
    @coro
    def checkNodeRS(self,*args):
        return True
    @coro
    def checkForInsert(self):
        pass

class Poset(Chare):

    @coro
    def __init__(self, peSpec, nodeConstructor, localVarGroup, successorChare, usePosetChecking, feederSpec):

        # self.stackNum = batchSize
        # To do: check to make sure we're passed a valid Group in localVarGroup
        self.usePosetChecking = usePosetChecking
        self.localVarGroup = localVarGroup
        self.useDefaultLocalVarGroup = False

        self.nodeConstructor = nodeConstructor
        if self.nodeConstructor is None:
            self.nodeConstructor = PosetNode
        self.successorChare = successorChare
        if successorChare is None:
            self.successorChare = successorWorker
        else:
            self.successorChare = successorChare
        # Create a group to paralellize the computation of successors
        # (Use all PEs unless a list was explicitly passed to us)
        if peSpec == None:
            self.posetPEs = [(0,charm.numPes(),1)]
            self.hashPEs = [(0,charm.numPes(),1)]
        else:
            self.posetPEs = peSpec['poset']
            self.hashPEs = peSpec['hash']

        self.posetPElist = list(itertools.chain.from_iterable( \
               [list(range(r[0],r[1],r[2])) for r in self.posetPEs] \
            ))
        self.hashPElist = list(itertools.chain.from_iterable( \
               [list(range(r[0],r[1],r[2])) for r in self.hashPEs] \
            ))
        self.succGroupFull = Group(self.successorChare,args=[])
        charm.awaitCreation(self.succGroupFull)

        if localVarGroup is None:
            self.useDefaultLocalVarGroup = True
            self.usePosetChecking = False
            self.localVarGroup = Group(localVar,args=[])
            charm.awaitCreation(self.localVarGroup)
            self.localVarGroup.init(self.succGroupFull,self.posetPElist,awaitable=True).get()

        # Create a PE scheduler Chare for use in Reverse Search implementations
        self.rsPeScheduler = Chare(peSchedulerRS,args=[self.succGroupFull,self.posetPElist],onPE=0)
        charm.awaitCreation(self.rsPeScheduler)

        self.succGroupFull.initPEs(self.posetPElist,localVarGroup=self.localVarGroup,rsScheduler=self.rsPeScheduler)
        secs = [self.succGroupFull[r[0]:r[1]:r[2]] for r in self.posetPEs]
        self.succGroup = charm.combine(*secs)
        successorProxies = self.succGroupFull.getProxies(ret=True).get()
        self.successorProxies = list(itertools.chain.from_iterable( \
                [successorProxies[r[0]:r[1]:r[2]] for r in self.posetPEs] \
            ))
        self.useGPU = False

        # Initialize a new distributed hash table:
        self.distHashTable = Chare(DistributedHash.DistHash,args=[
            self.succGroupFull, \
            self.nodeConstructor, \
            self.localVarGroup, \
            self.hashPEs, \
            self.posetPEs, \
            feederSpec \
        ],onPE=0)
        charm.awaitCreation(self.distHashTable)
        self.migrationInfo = {'poset':[(self.posetPElist,self.thisProxy), (self.posetPElist, self.rsPeScheduler)], 'hash':[(self.hashPElist,self.distHashTable)] + self.distHashTable.getMigrationInfo(ret=True).get()}
        # print('Initialized distHashTable group')
        self.oldFlippedConstraints = None
    @coro
    def init(self):
        initFut = self.distHashTable.initialize(awaitable=True)
        initFut.get()
    @coro
    def getStats(self):
        statsFut = Future()
        self.succGroupFull.getStats(statsFut)
        stats = statsFut.get()
        stats['RegionCount'] = sum(self.levelSizes) + stats['RSRegionCount']
        stats['LevelSizes'] = self.levelSizes
        return stats
    @coro
    def getMigrationInfo(self):
        return self.migrationInfo

    def initialize(self, AbPairs, pt, fixedA, fixedb, normalize=1.0):
        self.AbPairs = deepcopy(AbPairs)
        self.pt = pt
        self.fixedA = fixedA.copy()
        self.fixedb = fixedb.copy()
        self.normalize = normalize
        if not self.normalize is None and not ( isinstance(self.normalize,float) and self.normalize > 0.0 ):
            raise ValueError('ERROR: normalize argument must be \'None\' or a float > 0')
        # self.N = len(self.AbPairs[0][0])


    @coro
    def setConstraint(self,lb=0,out=0,timeout=None,prefilter=True,rebasePt=None):
        self.populated = False
        self.incomplete = True
        # self.N = len(self.AbPairs[0][0])
        if prefilter:
            createConstraints = region_helpers.flipConstraintsReducedMin
        else:
            createConstraints = region_helpers.flipConstraints
        self.flippedConstraints = createConstraints( \
                -1*self.AbPairs[out][0], \
                self.AbPairs[out][1] - lb, \
                self.pt, \
                self.fixedA, \
                self.fixedb, \
                normalize=self.normalize \
            )
        self.N = self.flippedConstraints.N
        if not rebasePt is None:
            self.flippedConstraints.setRebase(copy(rebasePt))


        stat = self.succGroup.initialize(self.N,self.flippedConstraints.serialize(),timeout,awaitable=True)
        stat.get()
        if self.useDefaultLocalVarGroup:
            self.localVarGroup.setConstraintsOnly(self.flippedConstraints.serialize(),awaitable=True).get()

        self.populated = False
        self.oldFlippedConstraints = None
        self.levelSizes = [0]

        return 1

    @coro
    def initAndSetFromConstraints(self,constraintsObj,timeout=None,rebasePt=None):
        if not isinstance(constraintsObj,region_helpers.flipConstraints):
            print(f'ERROR: must provide a region_helpers.flipConstraints object!')
            return 0
        self.populated = False
        self.incomplete = True
        self.flippedConstraints = deepcopy(constraintsObj)
        allConstraints = self.flippedConstraints.allConstraints
        allN = self.flippedConstraints.allN
        self.AbPairs = [[-allConstraints[:allN,1:].copy(), -allConstraints[:allN,0].reshape(-1,1).copy()]]
        self.fixedA = allConstraints[allN:,1:]
        self.fixedb = -allConstraints[allN:,0].reshape(-1,1)
        self.pt = self.flippedConstraints.pt

        self.normalize = self.flippedConstraints.normalize
        self.N = self.flippedConstraints.N
        if not rebasePt is None and self.flippedConstraints.rebasePt is None:
            self.flippedConstraints.setRebase(copy(rebasePt))


        stat = self.succGroup.initialize(self.N,self.flippedConstraints.serialize(),timeout,awaitable=True)
        stat.get()
        if self.useDefaultLocalVarGroup:
            self.localVarGroup.setConstraintsOnly(self.flippedConstraints.serialize(),awaitable=True).get()

        self.populated = False
        self.oldFlippedConstraints = None
        self.levelSizes = [0]

        return 1

    @coro
    def getConstraintsObject(self):
        return self.flippedConstraints.serialize()

    @coro
    def setSuccessorCommonProperty(self,prop,val):
        self.succGroupFull.setProperty(prop,val,awaitable=True).get()
    @coro
    def getSuccGroupProxy(self):
        return self.succGroupFull
    @coro
    def getTableLen(self):
        return self.distHashTable.getTableLen(ret=True).get()

    @coro
    def getHashTableProxy(self):
        return self.distHashTable
    @coro
    def clearHashTable(self):
        self.distHashTable.clearHashTable(awaitable=True).get()
    @coro
    def newTable(self,tableName):
        return self.distHashTable.newTable(tableName,ret=True).get()
    @coro
    def isTable(self,tableName):
        return self.distHashTable.isTable(tableName,ret=True).get()
    @coro
    def getActiveTable(self):
        return self.distHashTable.getActiveTable(ret=True).get()
    @coro
    def activateTable(self,tableName):
        return self.distHashTable.activateTable(tableName,ret=True).get()
    @coro
    def copyTable(self,src=None,dest=None):
        return self.distHashTable.copyTable(src=src,dest=dest,ret=True).get()
    @coro
    def deleteTable(self,tableName):
        return self.distHashTable.deleteTable(tableName,ret=True).get()
    @coro
    def getTableNames(self):
        return self.distHashTable.getTableNames(ret=True).get()

    # Because charm4py seems to filter **kwargs, pass all arguments to populatePoset in a single dictionary.
    # This avoids having to distinguish between those arguments that are for populatePoset itself and those
    # that are merely passed on to setMethod. This is an implementation distinction not a semantic one: all
    # of these arguments affect the behavior/output of "populatePoset"

    # opts dictionary keys 'clearTable' and 'retrieveFaces' set parameters in populatePoset itself; any
    # other keys are passed as keyword arguments to setMethod
    @coro
    def populatePoset(self,face=None,witness=None,adjUpdate=None,payload=None, opts={} ):
        if self.populated:
            return
        self.clearTable = 'speed'
        self.retrieveFaces = False
        self.verbose = True
        self.sendFaces = False
        self.queryReturnInfo = False
        self.maxLevels = self.N + 1
        defaultSettings = ['clearTable','retrieveFaces','verbose','sendFaces','queryReturnInfo','maxLevels']
        for ky in defaultSettings:
            if ky in opts:
                setattr(self,ky,opts[ky])
                #opts.pop(ky)
        if self.clearTable and self.sendFaces:
            print(f'ERROR: \'clearTable\' flag is incompatible with \'sendFaces\'.')
            return None


        #print(f'verbose is {self.verbose}')
        self.succGroup.setMethod(**opts)

        if 'hashStore' in opts and opts['hashStore'] == 'vertex':
            self.hashStoreMode = 2
            tol = opts['tol'] if 'tol' in opts else 1e-9
            rTol = opts['rTol'] if 'rTol' in opts else 1e-9
            self.distHashTable.updateNodeEqualityFn(nodeType='vertex', tol=tol, rTol=rTol, H=self.flippedConstraints.constraints, awaitable=True).get()
        else:
            self.distHashTable.updateNodeEqualityFn(nodeType='standard', awaitable=True).get()
            self.hashStoreMode = 0

        self.distHashTable.resetLevelCount(awaitable=True).get()
        #self.succGroup.testSend()

        checkVal = True
        level = 0
        thisLevel = [(self.flippedConstraints.root,tuple())]
        posetLen = 1
        self.levelSizes = [1]
        timedOut = False

        # Send this node into the distributed hash table and check it
        initFut = Future()
        self.distHashTable.initListening(initFut,queryReturnInfo=self.queryReturnInfo,awaitable=True).get()
        initFut.get()
        self.succGroupFull.startListening(awaitable=True).get()

        boolIdxNoFlip = bytearray(b'\x00') * (self.flippedConstraints.wholeBytes + (1 if self.flippedConstraints.tailBits != 0 else 0))
        for unflipIdx in range(len(thisLevel[0][0])-1,-1,-1):
            boolIdxNoFlip[thisLevel[0][0][unflipIdx]//8] = boolIdxNoFlip[thisLevel[0][0][unflipIdx]//8] | (1<<(thisLevel[0][0][unflipIdx] % 8))
        print(f'boolIdxNoFlip = {boolIdxNoFlip}')
        self.successorProxies[0].hashAndSend([ \
                                        boolIdxNoFlip, \
                                        thisLevel[0][0], \
                                        self.flippedConstraints.N, \
                                        tuple() if face is None else face, \
                                        self.flippedConstraints.pt if witness is None else witness \
                                    ], \
                                    adjUpdate=adjUpdate, \
                                    payload=(tuple() if payload is None else payload), \
                                    vertex=(None if self.hashStoreMode != 2 else (self.flippedConstraints.pt,tuple())), \
                                ret=True).get()
        thisLevel = [( \
                      boolIdxNoFlip if self.hashStoreMode == 0 else thisLevel[0][0], \
                      self.flippedConstraints.N, \
                      0, \
                      tuple() if face is None else face, \
                      self.flippedConstraints.pt if witness is None else witness, \
                      adjUpdate, \
                      tuple() if payload is None else payload
                    )]

        self.distHashTable.awaitPending(usePosetChecking=self.usePosetChecking, awaitable=True).get()
        # Send a final termination signal:
        self.succGroup.sendAll(-2,awaitable=True).get()
        self.succGroup.closeQueryChannels(awaitable=True).get()
        self.succGroup.flushMessages(ret=True).get()

        checkVal = self.distHashTable.levelDone(ret=True).get()
        if not checkVal:
            level = self.N+2
        listenerCount = self.distHashTable.awaitShutdown(ret=True).get()

        # If clearTable is set, then the result won't be the full table, so might as well
        # clear what we have in there already (I'm torn about the logic of this behavior...)
        if self.clearTable:
            self.distHashTable.clearHashTable(awaitable=True).get()

        doneFuts = [Future() for k in range(len(self.successorProxies))]
        for k in range(len(self.successorProxies)):
            self.successorProxies[k].initList( doneFuts[k] )
        cnt = 0
        for fut in charm.iwait(doneFuts):
            cnt += fut.get()

        iFut = Future()
        self.successorProxies[0].initListNew(thisLevel,iFut)
        iFut.get()
        nextLevelSize = 1

        # print('Waiting for level done')
        while level < self.maxLevels and nextLevelSize > 0:
            # successorProxies = self.succGroup.getProxies(ret=True).get()
            # doneFuts = [Future() for k in range(len(self.successorProxies))]
            # for k in range(len(self.successorProxies)):
            #     self.successorProxies[k].initListNew( \
            #                 [ i for i in thisLevel[k:len(thisLevel):len(self.posetPElist)] ], \
            #                 doneFuts[k]
            #             )
            # cnt = 0
            # for fut in charm.iwait(doneFuts):
            #     cnt += fut.get()

            initFut = Future()
            self.distHashTable.initListening(initFut,queryReturnInfo=self.queryReturnInfo,awaitable=True).get()
            initFut.get()
            self.succGroupFull.startListening(awaitable=True).get()

            if not self.useGPU:
                self.succGroup.computeSuccessorsNew(ret=True).get()
            else:
                self.succGroup.computeSuccessorsNewGPU(ret=True).get()
            timedOut = any(self.succGroupFull.getTimeout(ret=True).get())
            if timedOut:
                print('Received timeout on level ' + str(level))

            self.distHashTable.awaitPending(usePosetChecking=self.usePosetChecking, awaitable=True).get()
            self.succGroup.sendAll(-2,awaitable=True).get()
            self.succGroup.closeQueryChannels(awaitable=True).get()
            self.succGroup.flushMessages(ret=True).get()

            # print('Finished looking for successors on level ' + str(level))
            checkVal = self.distHashTable.levelDone(ret=True).get()
            if not checkVal or timedOut:
                if timedOut: checkVal = None
                break
            # print('Done with level ' + str(level))

            # Retrieve faces for all the nodes in the current level
            # print(nextLevelSize)
            if self.retrieveFaces:
                facesFuts = [Future() for _ in range(len(self.posetPElist))]
                for k in range(len(facesFuts)):
                    self.succGroupFull[self.posetPElist[k]].retrieveFaces(facesFuts[k])
                faces = {}
                for fut in charm.iwait(facesFuts):
                    retPe, facesList = fut.get()
                    faces[retPe] = facesList

            # nextLevel = self.distHashTable.getLevelList(ret=True).get()
            prevLevelSize = nextLevelSize

            nextLevelSize = self.distHashTable.scheduleNextLevel(clearTable=(self.clearTable == 'memory'),ret=True).get()
            self.levelSizes.append(nextLevelSize)

            listenerCount = self.distHashTable.awaitShutdown(ret=True).get()

            if self.clearTable == 'speed':
                self.distHashTable.clearHashTable(awaitable=True).get()


            posetLen += nextLevelSize
            # print(posetLen)
            if self.verbose:
                print(f'Finished level {level} of size {prevLevelSize}')


            # thisLevel = nextLevel
            level += 1

        # Note, this print has to go here because this coroutine is only suspending until checkNodes is set
        statsFut = Future()
        self.succGroupFull.getStats(statsFut)
        stats = statsFut.get()
        if self.verbose:
            stats['levelSizes'] = self.levelSizes
            print('Total LPs used: ' + str(stats))

            print('Checker returned value: ' + str(checkVal))

            # print('Computed a (partial) poset of size: ' + str(len(self.hashTable.keys())))
            print('Computed a (partial) poset of size: ' + str(posetLen))

        if timedOut:
            print('Poset computation timed out...')
        # return [i.iINT for i in self.hashTable.keys()]
        self.populated = True
        return checkVal

    @coro
    def insertHyperplane(self,newA, newb, normalize=1.0, opts={}):
        nrm = np.linalg.norm(np.hstack([-newA.flatten(), newb.flatten()])) / normalize
        newA = -newA.copy().flatten() / nrm
        newb = copy(newb).flatten() / nrm
        self.oldFlippedConstraints = deepcopy(self.flippedConstraints)

        inserted = self.flippedConstraints.insertHyperplane(newA, newb)
        self.pt = self.flippedConstraints.pt
        # To do?: have each instance insert the hyperplane separately for speed (avoids copying custom object)
        aug = self.flippedConstraints.deserialize()
        if not inserted:
            self.flippedConstraints.serialize()
            self.succGroup.initialize(aug.N, aug, None, awaitable=True).get()
            self.localVarGroup.setConstraintsOnly(aug.serialize(),awaitable=True).get()
            return True

        localOpts = deepcopy(opts)
        localOpts['method'] = 'insertHyperplane'
        localOpts['clearTable'] = False
        localOpts['sendWitness'] = True
        localOpts['sendFaces'] = True
        localOpts['queryReturnInfo'] = True
        localOpts['useQuery'] = False
        localOpts['maxLevels'] = aug.N + 2
        tol = opts['tol'] if 'tol' in opts else 1e-9
        rTol = opts['rTol'] if 'rTol' in opts else 1e-9
        solver = localOpts['solver'] if 'solver' in localOpts else 'glpk'
        lpopts = localOpts['lpopts'] if 'lpopts' in localOpts else None

        # From now on, we are going to work in CDD format...
        hyper = np.hstack([-newb, newA])
        print(f'Old hyper = {hyper}')
        hyper = aug.constraints[aug.N-1,:]
        print(f'New hyper = {hyper}')

        projWb, subIdx = region_helpers.projectConstraints(aug.constraints[:aug.N-1,:],hyper,tol=tol,rTol=rTol)
        projFixed, _ = region_helpers.projectConstraints(aug.constraints[aug.N:,:],hyper,subIdx=subIdx,tol=tol,rTol=rTol)

        if projWb.shape[1] > 1: # Should be equivalent to self.tll.n >= 2
            pt = region_helpers.findInteriorPoint(projFixed,solver=solver,tol=tol,rTol=rTol,lpopts=lpopts)
            hyperSlice = np.hstack([-hyper[1:(subIdx+1)],-hyper[(subIdx+2):]]).reshape(-1,1)
            is1d = False
        else:
            pt = np.array([[hyper[0]/-hyper[1]]],dtype=np.float64)
            if not np.all(projFixed[:,0] >= -self.tol):
                pt = None
            hyperSlice = np.array([[0]],dtype=np.float64)
            is1d = True

        if pt is None:
            print(f'Inserted hyperplane doesn\'t intersect constraint set...')
            return False

        print(hyper[1:].shape)
        print(pt)

        ptLift = np.zeros((hyper.shape[0]-1,1),dtype=np.float64)
        ptLift[:subIdx,0] = pt[:subIdx,0]
        ptLift[(subIdx+1):,0] = pt[subIdx:,0]
        ptLift[subIdx,0] = (-1/hyper[subIdx+1]) * (-(hyperSlice.reshape(1,-1) @ pt)[0,0] + hyper[0])

        # This will create a constraints object for the projected constraints
        # (Notably: with duplicates hidden by the new functionality)
        projFlipConstraints = region_helpers.flipConstraintsReducedMin( \
                                            projWb[:,1:], \
                                            -projWb[:,0].reshape(-1,1), \
                                            pt, \
                                            fA = projFixed[:,1:], \
                                            fb = -projFixed[:,0].reshape(-1,1), \
                                            tol = tol, \
                                            rTol = rTol, \
                                            normalize = 1.0 \
                                        )

        print(np.abs(-hyper[1:].reshape(1,-1) @ ptLift - hyper[0]))

        newBaseRegFullTup = tuple(np.nonzero((-aug.constraints[:(aug.N-1),1:] @ ptLift - aug.constraints[:(aug.N-1),0].reshape(-1,1)).flatten() >= tol)[0])
        newBaseRegFull = region_helpers.tupToBytes(newBaseRegFullTup, *region_helpers.byteLenFromN(aug.N))
        rebasePt = region_helpers.findInteriorPoint(aug.getRegionConstraints(newBaseRegFullTup),solver=solver,tol=tol,rTol=rTol,lpopts=lpopts)
        if rebasePt is None:
            print(f'WARNING: Should never happen!!!')
            rebasePt = region_helpers.findInteriorPoint(aug.getRegionConstraints(newBaseRegFullTup + (aug.N-1,)),solver=solver,tol=tol,rTol=rTol,lpopts=lpopts)
            newBaseRegFullTup = newBaseRegFullTup + (aug.N-1,)
        if rebasePt is None:
            print(f'Couldn\'t find a new rebase point')
            self.flippedConstraints = self.oldFlippedConstraints
            self.oldFlippedConstraints = None
            return False

        print(newBaseRegFullTup)
        print(rebasePt)

        print(np.allclose(aug.getRegionConstraints(tuple()) , aug.constraints))

        # Now we query the poset to find the correct encoding for the region we care about
        # This will be sent with adjUpdate[-1] to bootstrap the process of removing the old node encodings
        f = Future()
        self.distHashTable.initListening(f,queryReturnInfo=True,awaitable=True).get()
        f.get()
        self.succGroup.startListening(awaitable=True).get()

        print(f' base N = {aug.baseN} aug.N = {aug.N}')

        # Identify the correct encoding of the first region split by the inserted hyperplane (that identified
        # by the point ptLift). That is, determine how many hyperplanes were used to encode this region in the
        # hash table.
        for idx in range(1,aug.N - aug.baseN + 1):
            print(f'    <><><><>    Trying  to recode {newBaseRegFull} to length {region_helpers.byteLenFromN(aug.N-idx)}')
            newBaseReg, newBaseRegTup, newBaseRegN = region_helpers.recodeRegNewN(-idx, newBaseRegFull, aug.N)
            print((newBaseReg, newBaseRegTup, newBaseRegN))
            retVal = self.succGroup[0].query([newBaseReg, newBaseRegTup, newBaseRegN],ret=True).get()
            print(retVal)
            if retVal[0] > 0:
                stripNum = idx
                break

        aug.root = tuple(newBaseRegFullTup)
        aug.setRebase(rebasePt)

        newAdj = deepcopy(retVal[3])
        print(f',,,,,,   newAdj = {newAdj}')
        newAdj[-1] = aug.N - stripNum
        for f in retVal[1]:
            if not f in newAdj:
                newAdj[f] = self.flippedConstraints.baseN
        print(newAdj)

        # Insertion
        newBaseRegFullTup = tuple( \
                                np.nonzero( \
                                    ( \
                                        -self.flippedConstraints.constraints[:(self.flippedConstraints.N-1),1:] @ rebasePt \
                                        - self.flippedConstraints.constraints[:(self.flippedConstraints.N-1),0].reshape(-1,1) \
                                    ).flatten() \
                                    >= tol \
                                )[0] \
                            )
        print(f'    """"" newBaseRegFullTup = {newBaseRegFullTup}')
        boolIdxNoFlipFull, INTrepFull, _ = region_helpers.recodeRegNewN(0, newBaseRegFullTup, aug.N)
        boolIdxNoFlip, INTrep, _ = region_helpers.recodeRegNewN( -stripNum, INTrepFull, aug.N)
        print(f'{self.flippedConstraints.rebaseRegion(INTrepFull)}')
        rebasedINTrep = region_helpers.recodeRegNewN(-stripNum,self.flippedConstraints.rebaseRegion(INTrepFull)[0],aug.N)[1]
        rebasedINTrepSet = set(rebasedINTrep)

        rebasedINTrep = region_helpers.recodeRegNewN( -stripNum ,self.flippedConstraints.rebaseRegion(INTrepFull)[0],aug.N)[1]
        rebasedINTrepSet = set(rebasedINTrep)
        print(f'///// rebasedINTrep = {rebasedINTrep}; INTrep = {INTrep}; boolIdxNoFlip = {boolIdxNoFlip}, boolIdxNoFlipFull = {boolIdxNoFlipFull}')

        # Initialize the adj dict's of nodes if this is the first inserted hyperplane
        if aug.N == aug.baseN + 1:
            self.distHashTable.tableApplyMethod('adjFaceCreate',awaitable=True).get()

        # We have to retrieve the information from the node in the table that is going to be split
        # q = self.succGroup[0].query( [boolIdxNoFlip, INTrep, aug.N - stripNum], op=DistributedHash.QUERYOP_DELETE, ret=True).get()
        # if q[0] > 0:
        #     oldFace = q[1]
        #     oldWitness = q[2]
        #     oldAdj = q[3]
        #     oldPayload = q[4]
        # else:
        #     print(f'Root node not found!')
        #     return
        # newAdj[-1] = aug.N
        # print(f'()()()    q = {q}')
        # cont = self.succGroup[0].hashAndSend( \
        #                 ( \
        #                     boolIdxNoFlipFull, \
        #                     INTrepFull, \
        #                     aug.N \
        #                 ) + ( \
        #                     set(), \
        #                     rebasePt \
        #                 ), \
        #                 adjUpdate={-1:aug.N-stripNum}, \
        #                 payload = None, \
        #                 ret=True \
        #             ).get()

        self.distHashTable.awaitPending(usePosetChecking=False,awaitable=True).get()
        self.succGroup.sendAll(-2,awaitable=True).get()
        self.succGroup.closeQueryChannels(awaitable=True).get()
        self.succGroup.flushMessages(ret=True).get()

        self.succGroup.initialize(aug.N, aug.serialize(), None, awaitable=True).get()
        self.localVarGroup.setConstraintsOnly(aug.serialize(),awaitable=True).get()

        self.succGroup.setProperty('iHyper', hyper, awaitable=True).get()
        self.succGroup.setProperty('iFlipConstraints', projFlipConstraints.serialize(), awaitable=True).get()
        self.succGroup.setProperty('iSubIdx', subIdx, awaitable=True).get()
        self.succGroup.setProperty('iIs1d', is1d, awaitable=True).get()
        self.succGroup.setProperty('iPtLift', ptLift, awaitable=True).get()

        self.distHashTable.setCheckDispatch({'check':'checkForInsert','update':'updateForInsert'},awaitable=True).get()

        self.populated = False
        self.thisProxy.populatePoset(face=set(),witness=rebasePt,adjUpdate={-1:aug.N-stripNum},payload=deepcopy(retVal[4]),opts=localOpts,awaitable=True).get()

        # Now that we're all done, restore default dispatch for check/update
        self.distHashTable.setCheckDispatch({'check':'check','update':'update'},awaitable=True).get()

    @coro
    def canonicalizeTable(self,opts={},rebasePt=None):

        localOpts = deepcopy(opts)
        localOpts['method'] = 'canonicalizeTable'
        localOpts['clearTable'] = False
        localOpts['sendWitness'] = True
        localOpts['sendFaces'] = True
        localOpts['queryReturnInfo'] = True
        localOpts['useQuery'] = False
        tol = opts['tol'] if 'tol' in opts else 1e-9
        rTol = opts['rTol'] if 'rTol' in opts else 1e-9
        solver = localOpts['solver'] if 'solver' in localOpts else 'glpk'
        lpopts = localOpts['lpopts'] if 'lpopts' in localOpts else None

        self.distHashTable.setCheckDispatch({'check':'checkForInsert','update':'updateForInsert'},awaitable=True).get()

        tval = self.distHashTable.seedLevelFullTable(clearTable=True,ret=True).get()

        self.succGroup.setMethod(**localOpts)

        self.succGroup.setProperty('iPtLift', rebasePt, awaitable=True).get()
        self.succGroup.setProperty('useRebase', False if rebasePt is None else True, awaitable=True).get()

        ##### Now test the canonicalization features...
        f = Future()
        self.distHashTable.initListening(f,queryReturnInfo=True,awaitable=True).get()
        f.get()
        self.succGroup.startListening(awaitable=True).get()

        self.succGroup.computeSuccessorsNew(ret=True).get()

        self.distHashTable.awaitPending(usePosetChecking=False,awaitable=True).get()
        self.succGroup.sendAll(-2,awaitable=True).get()
        self.succGroup.closeQueryChannels(awaitable=True).get()
        self.succGroup.flushMessages(ret=True).get()
        #####
        self.distHashTable.setCheckDispatch({'check':'check','update':'update'},awaitable=True).get()

    @coro
    def getLPCount(self):
        statsFut = Future()
        self.succGroupFull.getStats(statsFut)
        stats = statsFut.get()
        return stats

    @coro
    def populatePosetRS(self,payload=None, opts={}):
        if self.populated:
            return
        self.verbose = True
        defaultSettings = ['verbose']
        for ky in defaultSettings:
            if ky in opts:
                setattr(self,ky,opts[ky])

        opts['reverseSearch'] = True
        self.succGroup.setMethod(**opts)
        self.rsPeScheduler.resetScheduler(verbose=self.verbose,awaitable=True).get()

        checkVal = True
        level = 0
        thisLevel = [(self.flippedConstraints.root,)]
        posetLen = 1
        timedOut = False

        # Start reverse search on the root on the first PE
        peToUse = self.rsPeScheduler.schedNextFreePE(ret=True).get()
        if peToUse >= 0:
            self.succGroup[peToUse].reverseSearch(self.flippedConstraints.root,payload=(tuple() if payload is None else payload),witness=self.flippedConstraints.pt)
        else:
            print('Error: RS Pe scheduler not configured properly')

        checkVal = self.rsPeScheduler.awaitResult(awaitable=True).get()

        statsFut = Future()
        self.succGroupFull.getStats(statsFut)
        stats = statsFut.get()
        regionDist = self.succGroup.getProperty('rsRegionCount',ret=True).get()

        if self.verbose:
            print('Total LPs used: ' + str(stats))

            print('Checker returned value: ' + str(checkVal))

            # print('Computed a (partial) poset of size: ' + str(len(self.hashTable.keys())))
            print('Computed a (partial) poset of size: ' + str(stats['RSRegionCount']))

            print(f'Regions discovered by PE: {regionDist}')

            if timedOut:
                print('Poset computation timed out...')
        # return [i.iINT for i in self.hashTable.keys()]
        self.populated = True
        return checkVal


class peSchedulerRS(Chare):

    def __init__(self, successorGroup, posetPElist):
        self.succGroup = successorGroup
        self.posetPElist = posetPElist
        self.peFree = copy(self.posetPElist)
        self.resultFut = None
        self.retVal = True
        self.verbose = True

    @coro
    def resetScheduler(self,verbose):
        self.verbose = verbose
        self.peFree = copy(self.posetPElist)
        self.succGroup.setPeAvailableRS(True,awaitable=True).get()
        self.resultFut = None
        self.retVal = True

    @coro
    def awaitResult(self):
        self.resultFut = Future()
        return self.resultFut.get()

    # If there is a PE available, schedule it. If there isn't, return -1 so the successorWorker
    # knows that the scheduling was unsuccessful (as my be the case due to concurrency issues)
    @coro
    def schedNextFreePE(self):
        if not self.retVal:
            return -1
        if len(self.peFree) > 1:
            return self.peFree.pop()
        elif len(self.peFree) == 1:
            lastFreePe = self.peFree.pop()
            self.succGroup.setPeAvailableRS(False,abort=(not self.retVal))
            return lastFreePe
        else:
            return -1

    @coro
    def freePe(self,pe):
        # print(f'Free PEs = {self.peFree}; To free = {pe}; retVal = {self.retVal}')
        self.peFree.append(pe)
        if len(self.peFree) == len(self.posetPElist):
            if self.verbose:
                print(f'*** All done! {self.peFree} ***')
            self.resultFut.send(self.retVal)
        self.succGroup.setPeAvailableRS(self.retVal,abort=(not self.retVal))

    @coro
    def failAbort(self):
        self.retVal = False
        self.succGroup.setPeAvailableRS(False,abort=True)

    # @coro
    # def setrsDone(self):
    #     self.succGroup.setrsDone(awaitable=True).get()



class successorWorker(Chare):

    def initPEs(self,pes,localVarGroup=None,rsScheduler=None):
        self.posetPElist = pes
        self.checkRS= False
        test = getattr(localVarGroup,'checkNodeRS',None)
        if not test is None and callable(localVarGroup.checkNodeRS):
            self.checkRS = True
        self.doRSCleanup = False
        test = getattr(self,'cleanupRS',None)
        if not test is None and callable(self.cleanupRS):
            self.doRSCleanup = True
        self.localVarGroup = localVarGroup
        self.timedOut = False
        self.method = None
        self.rsScheduler = rsScheduler
        self.rsPeFree = True
        self.rsDone = False
        self.rsDepth = 0
        self.rsRegionCount = 0
        self.sendFaces = False
        self.sendWitness = False
        self.deferLock = False
        self.hashedNodeCount = 0
        self.iHyper = None
        self.iFlipConstraints = None
        self.iSubIdx = None
        self.iIs1d = None
        self.iPtLift = None

    def initialize(self,N,constraints,timeout):
        self.workInts = []
        self.N = N
        self.flippedConstraints = constraints.deserialize()
        self.constraints = self.flippedConstraints.constraints
        # self.processNodeSuccessors = partial(successorWorker.processNodeSuccessorsFastLP, self, solver='glpk')
        self.processNodeSuccessors = self.thisProxy[self.thisIndex].processNodeSuccessorsFastLP
        self.processNodesArgs = {'solver':'glpk','ret':True}
        self.method = None
        # Defaults to glpk, so this empty call is ok:
        d = self.constraints.shape[1]-1
        self.lpObjs = {d:encapsulateLP.encapsulateLP(), d+1: encapsulateLP.encapsulateLP()}
        self.lp = self.lpObjs[d]
        self.lpIntPoint = self.lpObjs[d+1]
        if d > 1:
            self.lpObjs[d-1] = encapsulateLP.encapsulateLP()
        self.rsLP = encapsulateLP.encapsulateLP()
        self.rsLPIntPoint = encapsulateLP.encapsulateLP()
        # self.hashChannels = []
        self.clockTimeout = (timeout + time.time()) if timeout is not None else None
        self.timedOut = False
        self.stats = {'LPSolverCount':0, 'xferTime':0, 'numQueries':0, 'successfulQueries':0, 'RSRegionCount':0, 'RSLPCount':0}
        self.rsPeFree = True
        self.rsDone = False
        self.rsDepth = 0
        self.rsRegionCount = 0
        self.iHyper = None
        self.iFlipConstraints = None
        self.iSubIdx = None
        self.iIs1d = None
        self.iPtLift = None
    @coro
    def getTimeout(self):
        return self.timedOut

    def setMethod(self,method='fastLP',solver='glpk',useQuery=False,lpopts={},reverseSearch=False,hashStore='bits',tol=1e-9,rTol=1e-9,sendFaces=False,clearTable='speed',sendWitness=False,verbose=True):
        for d in self.lpObjs.keys():
            self.lpObjs[d].initSolver(solver=solver, opts={'dim':(d)})
        self.rsLP.initSolver(solver=solver, opts={'dim':(self.constraints.shape[1]-1)})
        self.rsLPIntPoint.initSolver(solver=solver, opts={'dim':(self.constraints.shape[1])})
        self.useQuery = useQuery
        self.doRS = reverseSearch
        self.tol = tol
        self.rTol = rTol
        self.sendFaces = sendFaces
        self.clearTable = clearTable
        self.sendWitness = True if sendWitness else None
        self.verbose = verbose
        self.method = method
        if hashStore == 'bits':
            self.hashStoreMode = 0
        elif hashStore == 'list':
            self.hashStoreMode = 1
        elif hashStore == 'vertex':
            self.hashStoreMode = 2
        else:
            self.hashStoreMode = 0
        if method=='cdd':
            self.processNodeSuccessors = self.thisProxy[self.thisIndex].processNodeSuccessorsCDD
            self.processNodesArgs = {'solver':solver}
            if self.hashStoreMode == 2:
                print(f'WARNING: vertex region encodings are not supported for method {method}. Defaulting to bit region encodings...')
        elif method=='fastLP':
            self.processNodeSuccessors = self.thisProxy[self.thisIndex].processNodeSuccessorsFastLP
            self.processNodesArgs = {'solver':solver}
            if self.hashStoreMode == 2:
                print(f'WARNING: vertex region encodings are not supported for method {method}. Defaulting to bit region encodings...')
        elif method=='insertHyperplane':
            self.processNodeSuccessors = self.thisProxy[self.thisIndex].processNodeSuccessorsInsertHyperplane
            self.processNodesArgs = {'solver':solver}
        elif method=='canonicalizeTable':
            self.processNodeSuccessors = self.thisProxy[self.thisIndex].processNodeSuccessorsCanonicalizeTable
            self.processNodesArgs = {'solver':solver}
        if len(lpopts) == 0:
            self.lpopts = {}
        else:
            self.lpopts = deepcopy(lpopts)
        if solver != 'glpk' and not 'fallback' in self.lpopts:
            self.lpopts['fallback'] = {'solver':'glpk'}
        self.lpopts['solver'] = solver
        self.processNodesArgs['lpopts'] = self.lpopts
        self.processNodesArgs['ret'] = True
        self.method = method
        self.solver = solver

        self.Hcol0Close = self.tol + self.rTol * np.abs(self.constraints[:,0])
        self.Hcol0CloseVertex = self.constraints[:,0] - self.Hcol0Close

    @coro
    def setProperty(self,prop,val):
        setattr(self,prop,val)

    @coro
    def getProperty(self,prop):
        return getattr(self,prop)

    @coro
    def setPeAvailableRS(self,status,abort=False):
        self.rsPeFree = status
        if abort:
            self.rsDone = True

    @coro
    def getStats(self, statsFut):
        if charm.myPe() in self.posetPElist:
            self.stats['LPSolverCount'] = sum([self.lpObjs[di].lpCount for di in self.lpObjs.keys()])
            self.stats['RSRegionCount'] = self.rsRegionCount
            self.stats['RSLPCount'] = self.rsLP.lpCount + self.rsLPIntPoint.lpCount
        retVal = defaultdict(int) if not charm.myPe() in self.posetPElist else self.stats
        self.reduce(statsFut,retVal,DictAccum)

    @coro
    def getProxies(self):
        return self.thisProxy[self.thisIndex]
    @coro
    def initHashChannel(self, procGroupProxies):
        if not charm.myPe() in self.posetPElist:
            return
        self.numHashWorkers = len(procGroupProxies)
        self.hashChannels = [Channel(self, remote=proxy) for proxy in procGroupProxies]
        self.numHashBits = 1
        while self.numHashBits < self.numHashWorkers:
            self.numHashBits = self.numHashBits << 1
        self.hashMask = self.numHashBits - 1
        self.numHashBits -= 1
        # if self.N % 4 == 0:
        #     self.numBytes = self.N/4
        # else:
        #     self.numBytes = int(self.N/4)+1
        # print(self.hashChannels)

    @coro
    def initQueryChannel(self, procGroupProxies, distHashProxy):
        if not charm.myPe() in self.posetPElist:
            return
        # self.numHashWorkers = len(procGroupProxies)
        self.queryChannels = [Channel(self, remote=proxy) for proxy in procGroupProxies]
        self.queryMutexChannel = None
        if not self.rateChannel is None:
            self.queryMutexChannel = Channel(self, remote=distHashProxy)
        # self.numHashBits = 1
        # while self.numHashBits < self.numHashWorkers:
        #     self.numHashBits = self.numHashBits << 1
        # self.hashMask = self.numHashBits - 1
        # self.numHashBits -= 1
        # if self.N % 4 == 0:
        #     self.numBytes = self.N/4
        # else:
        #     self.numBytes = int(self.N/4)+1
        # print(self.hashChannels)

    @coro
    def closeQueryChannels(self):
        if not charm.myPe() in self.posetPElist:
            return
        for ch in self.queryChannels:
            ch.send(-2)
        if not self.queryMutexChannel is None:
            self.queryMutexChannel.send(-2)

    @coro
    def initRateChannel(self,overlapPElist ):
        self.rateChannel = None
        self.overlapPElist = overlapPElist
        if not charm.myPe() in self.posetPElist:
            return
        if charm.myPe() in overlapPElist:
            self.rateChannel = Channel(self,remote=overlapPElist[charm.myPe()][1])
        # self.feedbackChannel = Channel(self,remote=proxy)
    @coro
    def testSend(self):
        for k in range(self.numHashWorkers):
            #print('Sending on to ' + str(k))
            #print(self.hashChannels[k])
            self.hashChannels[k].send((self.thisIndex,k))
            #print('Message sent!')

    def startListening(self):
        self.hashedNodeCount = 0
        if not charm.myPe() in self.posetPElist:
            return
        for ch in self.hashChannels:
            ch.send(-100)

    def hashNode(self,toHash,payload=None,vertex=None,adjUpdate=None):
        # hashInt = int(posetFastCharm_numba.hashNodeBytes(np.array(toHash[0],dtype=np.uint8)))
        # hashInt = hashNodeBytes(np.array(toHash[0],dtype=np.uint8))
        hashInt = hashNodeBytes(toHash[0])
        if self.hashStoreMode == 0:
            regEncode = toHash[0]
        elif self.hashStoreMode == 1:
            regEncode = tuple(toHash[1])
        elif self.hashStoreMode == 2 and vertex is not None:
            regEncode = vertex
        else:
            # default to tuple mode
            regEncode = tuple(toHash[1])
        N = toHash[2]
        if len(toHash) >= 4:
            face = toHash[3]
        else:
            face = tuple()
        if len(toHash) >= 5:
            witness = toHash[4]
        else:
            witness = None
        if payload is not None:
            return ( (hashInt & self.hashMask) % self.numHashWorkers , hashInt >> self.numHashBits, regEncode, N, charm.myPe(), face, witness, adjUpdate, payload)
        else:
            return ( (hashInt & self.hashMask) % self.numHashWorkers , hashInt >> self.numHashBits, regEncode, N, charm.myPe(), face, witness, adjUpdate )

    @coro
    def hashAndSend(self,toHash,payload=None,vertex=None,adjUpdate=None):
        self.hashedNodeCount += 1
        val = self.hashNode(toHash,payload=payload,vertex=vertex,adjUpdate=adjUpdate)
        self.hashChannels[val[0]].send(val)
        # print('Trying to hash integer ' + str(nodeInt))
        # retVal = self.thisProxy[self.thisIndex].deferControl(code=5,ret=True).get()
        retVal = self.thisProxy[self.thisIndex].deferControl(ret=True).get()
        # print('Saw defercontrol return the following within HashAndSend ' + str(retVal))
        return retVal
    @coro
    def getHashedNodeCount(self):
        return self.hashedNodeCount
    @coro
    def resetHashedNodeCount(self):
        self.hashedNodeCount = 0
    @coro
    def decHashedNodeCount(self):
        self.hashedNodeCount -= 1

    def decodeRegionStore(self,INTrep):
        if type(INTrep) == tuple and len(INTrep) == 2 and type(INTrep[1]) is tuple:
            incommingINTrep = INTrep
            Hsol = (-self.constraints[:,1:] @ INTrep[0]).flatten()
            flipIdxs = (Hsol  > self.Hcol0CloseVertex).flatten().astype(np.bool8)
            # print(flipIdxs)
            intersectionIdxs = np.nonzero((np.abs(Hsol - self.constraints[:,0]) <= self.Hcol0Close).flatten())[0]
            # print(intersectionIdxs)
            flipIdxs[intersectionIdxs] = np.zeros(intersectionIdxs.shape,dtype=np.bool8)
            flipIdxs[list(INTrep[1])] = np.ones(len(INTrep[1]),dtype=np.bool8)
            INTrep = tuple(np.nonzero(flipIdxs)[0])
            # print(f'{INTrep}==>{(incommingINTrep[0].flatten().tolist(),incommingINTrep[1])}')
        if type(INTrep) == tuple:
            intIdxNoFlip = list(INTrep)
            boolIdxNoFlip = tupToBytes(INTrep, self.flippedConstraints.wholeBytes, self.flippedConstraints.tailBits)
            intIdx = list(range(self.N))
            # boolIdx[-1] = boolIdx[-1] & ((1<<(self.tailBits+1))-1)
            for unflipIdx in range(len(INTrep)-1,-1,-1):
                intIdx.pop(INTrep[unflipIdx])
            # boolIdxNoFlip = np.full(self.N,False,dtype=bool)
            # boolIdxNoFlip[INTrep,] = np.full(len(INTrep),True,dtype=bool)
            # intIdx = np.where(boolIdxNoFlip==0)[0]
            # boolIdxNoFlip = np.packbits(boolIdxNoFlip,bitorder='little')
        elif type(INTrep) == bytearray:
            boolIdxNoFlip = INTrep
            INTrep = bytesToList(boolIdxNoFlip, self.flippedConstraints.wholeBytes, self.flippedConstraints.tailBits)
            intIdxNoFlip = INTrep
            INTrep = tuple(intIdxNoFlip)
            intIdx = list(range(self.N))
            # boolIdx[-1] = boolIdx[-1] & ((1<<(self.tailBits+1))-1)
            for unflipIdx in range(len(INTrep)-1,-1,-1):
                intIdx.pop(INTrep[unflipIdx])

        return INTrep, boolIdxNoFlip, intIdx, intIdxNoFlip

    @coro
    def deferControl(self, code=1):
        if not self.rateChannel is None:
            while self.deferLock:
                suspendFut = Future()
                suspendFut.send(1)
                suspendFut.get()
            self.deferLock = True
            self.rateChannel.send(code)
            control = self.rateChannel.recv()
            while control > 0:
                control = self.rateChannel.recv()
            if control == -3:
                self.deferLock = False
                return False
        self.deferLock = False
        return True

    @coro
    def query(self, q, op=None, tabIdx=-1):
        qOp = 0
        if isinstance(op,int):
            qOp = op
        # print('PE' + str(charm.myPe()) + ' Query to send is ' + str(q))
        self.stats['numQueries'] += 1
        val = self.hashNode(q)
        self.queryChannels[val[0]].send((qOp,tabIdx) + val)
        # print('PE' + str(charm.myPe()) + ' sending query ' + str(val))
        if not self.queryMutexChannel is None:
            self.queryMutexChannel.send(charm.myPe())
            # print('Waiting for query mutex on PE ' + str(charm.myPe()))
            self.queryMutexChannel.recv()
            # print('Received query mutex on PE ' + str(charm.myPe()))
            if charm.myPe() == val[0]:
                self.thisProxy[self.thisIndex].deferControl(code=3,ret=True).get()
                # print('Got Control Back from self query')
            else:
                self.thisProxy[self.thisIndex].deferControl(code=4,ret=True).get()
            # print('Got Control Back.')
            self.queryMutexChannel.send(1)
        retVal = self.queryChannels[val[0]].recv()
        # print('^^^^^^ Received answer to query ' + str(q) + ' of ' + str(retVal))
        if retVal[0] > 0:
            self.stats['successfulQueries'] += 1
        return retVal

    @coro
    def tester(self):
        print('Entered tester on PE ' + str(charm.myPe()))
        return charm.myPe()
    # @coro
    # def initList(self,workInts):
    #     self.status = Future()
    #     self.workInts = workInts
    #     self.status.send(1)

    @coro
    def initListNew(self,workInts, fut):
        self.workInts = workInts
        # print(self.workInts)
        fut.send(1)

    @coro
    def initList(self,fut):
        self.workInts = []
        fut.send(1)

    @coro
    def appendToWorkList(self,li,fut):
        self.workInts.extend(li)
        fut.send(len(li))

    #@coro
    def sendAll(self,val):
        if not charm.myPe() in self.posetPElist:
            return
        for ch in self.hashChannels:
            ch.send(val)

    @coro
    def flushMessages(self):
        if not charm.myPe() in self.overlapPElist:
            return
        self.rateChannel.send(2)



    @coro
    def computeSuccessorsNew(self):
        term = False
        if len(self.workInts) > 0:
            successorList = [[None,None,None,None,None,None,None] for k in range(len(self.workInts))]
            for ii in range(len(successorList)):
                successorList[ii] = self.processNodeSuccessors(self.workInts[ii][0],self.workInts[ii][1],self.flippedConstraints.constraints,**self.processNodesArgs,witness=self.workInts[ii][4], payload=self.workInts[ii][6],xN=self.workInts[ii][1],face=self.workInts[ii][3],adj=self.workInts[ii][5], awaitable=True).get()
                self.timedOut = (time.time() > self.clockTimeout) if self.clockTimeout is not None else False
                # print('Working on ' + str(self.workInts[ii]) + 'on PE ' + str(charm.myPe()) + '; with timeout ' + str(self.timedOut))
                if type(successorList[ii][1]) is int or self.timedOut:
                    term = True
                    if self.timedOut:
                        successorList[ii][1] = -1
                    break
        else:
            successorList = [[set([]),-1]]

        # self.thisProxy[self.thisIndex].sendAll(-2 if not term else -3, awaitable=True).get()
        if term:
            self.thisProxy[self.thisIndex].sendAll(-3, awaitable=True).get()
        self.thisProxy[self.thisIndex].flushMessages(awaitable=True).get()


        self.workInts = [successorList[ii][1] for ii in range(len(successorList))]
        # successorList = [successorList[ii][0] for ii in range(len(successorList))]

    @coro
    def reverseSearch(self,INTrep,payload=None,witness=None):
        #print(f'PE {charm.myPe()}: working on {INTrep}')
        #print(f'PE {charm.myPe()} working on region {INTrep}')
        if self.rsDone:
            if self.rsDepth == 0:
                self.rsScheduler.freePe(charm.myPe(),awaitable=True).get()
            return

        self.rsDepth += 1

        if self.checkRS:
            # INTrep2, boolIdxNoFlip, intIdx, intIdxNoFlip = self.decodeRegionStore(INTrep)
            tempRetVal = self.localVarGroup[charm.myPe()].checkNodeRS(INTrep,payload=payload, witness=witness, ret=True).get()
            # print(f'{charm.myPe()} --> INTrep = {INTrep2}; boolIdxNoFlip = {boolIdxNoFlip}; intIdx = {intIdx}; intIdxNoFlip = {intIdxNoFlip}; witness = {witness}; retVal = {tempRetVal}')
            if not tempRetVal:
                self.rsDone = True
                self.rsPeFree = False
                self.rsDepth -= 1
                self.rsScheduler.failAbort()
                if self.rsDepth == 0:
                    self.rsScheduler.freePe(charm.myPe(),awaitable=True).get()
                return
        # Compute all of the adjacent nodes (from among the unflipped hyperplanes)
        H2 = self.constraints.copy()
        successorList, _, witnessList = self.processNodeSuccessors(INTrep,self.N,H2,**self.processNodesArgs,payload=payload,witness=witness).get()
        if type(witnessList) is list and len(witnessList) == len(successorList):
            findWitnessLocally = False
        else:
            #print(f'PE {charm.myPe()}: successorList = {successorList}; witnessList = {witnessList}')
            findWitnessLocally = True
            witnessList = []
        #print(f'PE {charm.myPe()}: successors of {INTrep} are {successorList}')
        self.rsRegionCount += 1
        #print(f'PE {charm.myPe()} working on region {INTrep}; found successors {successorList}')

        for ii in range(len(successorList)):
            # Put check for path to root here...
            H = self.constraints.copy()
            H[successorList[ii][1],:] = -H[successorList[ii][1],:]
            if findWitnessLocally:
                interiorPoint = region_helpers.findInteriorPoint(H,lpObj=self.rsLPIntPoint,lpopts=self.lpopts)
                witnessList.append(interiorPoint)
            else:
                interiorPoint = witnessList[ii]
            # If the ray connecting interiorPoint to the origin point doesn't pass through the current
            # face, then we shouldn't explore this region from *the current* region (another will count it)
            # This face is stored in the third position of an element of successorList
            if interiorPoint is None:
                print(f'PE {charm.myPe()}: Something went wrong for region {INTrep}')
            else:
                rayEval = (H[:,0] + (H[:,1:] @ interiorPoint).flatten()).flatten() / (-H[:,1:] @ (-interiorPoint + self.flippedConstraints.pt)).flatten()
                #print(rayEval)
                rayScalar = np.min( np.where( rayEval < 0, np.inf, rayEval ) )
                rayHit = interiorPoint + rayScalar * (self.flippedConstraints.pt - interiorPoint)

                H2[INTrep,:] = -H2[INTrep,:]
                currentRegionIsParent = np.all(-H2[:,1:] @ rayHit - H2[:,0].reshape(-1,1) <= self.tol + self.rTol * np.abs(H2[:,0].reshape(-1,1)))
                H2[INTrep,:] = -H2[INTrep,:]
                if currentRegionIsParent:
                    #print(f'PE {charm.myPe()}: Visiting {successorList[ii][1]}')
                    peToUse = -1
                    if self.rsPeFree and not self.rsDone:
                        peToUse = self.rsScheduler.schedNextFreePE(ret=True).get()
                    if peToUse >= 0:
                        self.thisProxy[peToUse].reverseSearch(successorList[ii][1],payload=successorList[ii][6],witness=interiorPoint)
                    else:
                        self.thisProxy[self.thisIndex].reverseSearch(successorList[ii][1],payload=successorList[ii][6],witness=interiorPoint,awaitable=True).get()
        if self.doRSCleanup:
            self.thisProxy[self.thisIndex].cleanupRS(successorList,witnessList,awaitable=True).get()
        self.rsDepth -= 1
        if self.rsDepth == 0:
            self.rsScheduler.freePe(charm.myPe(),awaitable=True).get()
        return



    @coro
    def retrieveFaces(self,fut):
        fut.send( (charm.myPe(), self.workInts) )

    @coro
    def processNodeSuccessorsCDD(self,INTrep,N,H2,solver='glpk'):
        H = copy(H2)
        # global H2
        # H = np.array(H2)
        # H = np.array(processNodeSuccessors.H)
        idx = 1
        for i in range(N):
            if INTrep & idx > 0:
                H[i] = -1*H[i]
            idx = idx << 1

        mat = cdd.Matrix(H,number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
        ret = mat.canonicalize()
        to_keep = sorted(list(frozenset(range(len(H))) - ret[1]))
        if len(ret[0]) > 0:
            orig_to_keep = to_keep
            # There is some degeneracy, which means CDD screwed up (numerical ill-conditioning?)
            # Hence, we will use a direct implementation to find a minimal H-Representation
            to_keep = self.concreteMinHRep(H,copyMat=False,solver=solver)
            if orig_to_keep != to_keep:
                print('Linear regions found? ' + ('YES' if len(ret[0])>0 else 'NO'))
                print('CDD-obtained to_keep was:')
                print(orig_to_keep)
                print('GLPK Simplex-based Minimal H-Representation yielded to_keep of:')
                print(to_keep)
        # Use this to keep track of the region's faces
        facesInt = 0
        for k in to_keep:
            facesInt = facesInt + (1 << k)

        successors = []
        for i in range(len(to_keep)):
            if to_keep[i] >= N:
                break
            idx = 1 << to_keep[i]
            if idx & INTrep <= 0:
                successors.append( \
                        INTrep + idx \
                    )
                cont = self.thisProxy[self.thisIndex].hashAndSend(INTrep + idx,ret=True).get()
                if not cont:
                    return [set(successors), -1]

        return [set(successors), facesInt]


    @coro
    def concreteMinHRep(self,H2,constraint_list_in,boolIdxNoFlip,intIdxNoFlip,intIdx,solver='glpk',interiorPoint=None):
        witnessList = []
        safe = False
        if len(intIdx) == 0:
            return [], []
        d = H2.shape[1] - 1
        if d in self.lpObjs:
            lpObj = self.lpObjs[d]
        else:
            lpObj = encapsulateLP.encapsulateLP()
            lpObj.initSolver(solver=solver, opts={'dim':(d)})
        if d+1 in self.lpObjs:
            lpObjInt = self.lpObjs[d+1]
        else:
            lpObjInt = encapsulateLP.encapsulateLP()
            lpObj.initSolver(solver=solver, opts={'dim':(d+1)})

        restricted = False if constraint_list_in is None else True

        # H2 should be a view into the CDD-formatted H matrix selected by taking boolIdx or intIdx rows thereof
        if interiorPoint is not None:
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
            # Add an extra suspension to allow processing of queries/hashes from other PEs
            f = Future()
            f.send(1)
            f.get()
            if self.useQuery:
                boolIdxNoFlip[intIdx[idx]//8] = boolIdxNoFlip[intIdx[idx]//8] | (1<<(intIdx[idx]%8))
                insertIdx = 0
                while insertIdx < len(intIdxNoFlip) and intIdxNoFlip[insertIdx] < intIdx[idx]:
                    insertIdx += 1
                temp = copy(intIdxNoFlip)
                temp.insert(insertIdx,intIdx[idx])
                # q = self.thisProxy[self.thisIndex].query( bytes(np.packbits(boolIdx,bitorder='little')), ret=True).get()
                q = self.thisProxy[self.thisIndex].query( [boolIdxNoFlip, tuple(temp), self.flippedConstraints.N], ret=True).get()
                boolIdxNoFlip[intIdx[idx]//8] = boolIdxNoFlip[intIdx[idx]//8] ^ (1<<(intIdx[idx]%8))
                # print('PE' + str(charm.myPe()) + ' Queried table with node ' + str(origInt) + ' and received reply ' + str(q))
                # If the node corresponding to the hyperplane we're about to flip is already in the table
                # then treat it as redundant and skip it (saving the LP)
                if q[0] > 0:
                    continue
            if interiorPoint is None:
                # Set the extra row to the negation of the pre-relaxed current constraint
                H[-1,:] = -H2[intIdx[idx],:]
                H[offsetIdx,0] += 1
                status, x = lpObj.runLP( \
                        H2[intIdx[idx],1:], \
                        -H[constraint_list,1:], \
                        H[constraint_list,0], \
                        lpopts = self.lpopts, \
                        msgID = str(charm.myPe()) \
                    )
                H[offsetIdx,0] -= 1

                if status != 'optimal' and (safe or status != 'primal infeasible') and status != 'dual infeasible':
                    print('********************  PE' + str(charm.myPe()) + ' WARNING!!  ********************')
                    print('PE' + str(charm.myPe()) + ': Infeasible or numerical ill-conditioning detected at node' )
                    print('PE ' + str(charm.myPe()) + ': RESULTS MAY NOT BE ACCURATE!!')
                    return [set([]), 0]
                if (safe and -H2[intIdx[idx],1:]@x < H2[intIdx[idx],0]) \
                    or (not safe and (status == 'primal infeasible' or np.all(-H2[intIdx[idx],1:]@x - H2[intIdx[idx],0] <= self.tol + self.rTol * np.abs(H[:,0].reshape(-1,1))))):
                    # inequality is redundant, so skip it
                    constraint_list[offsetIdx] = False
                else:
                    to_keep.append(idx)
            else:
                H[offsetIdx,:] = -H[offsetIdx,:]
                x = region_helpers.findInteriorPoint(H,solver=solver,lpObj=lpObjInt,tol=self.tol,rTol=self.rTol,lpopts=self.lpopts)
                H[offsetIdx,:] = -H[offsetIdx,:]
                if x is not None:
                    # If x satisfies all of the original constraints then it is a redundant hyperplane
                    # intersecting with at least d other hyperplanes
                    notAdjacent = np.all(-H[:,1:] @ x - H[:,0].reshape(-1,1) <= self.tol + self.rTol * np.abs(H[:,0].reshape(-1,1)))
                    if notAdjacent:
                        constraint_list[offsetIdx] = False
                    else:
                        to_keep.append(idx)
                        witnessList.append(x)
                else:
                    constraint_list[offsetIdx] = False
        if restricted:
            # We are not solving full LPs, so the witness points aren't meaningful...
            return to_keep, []
        else:
            return to_keep, witnessList

    @coro
    def processNodeSuccessorsFastLP(self,INTrep,N,H,payload=[],solver='glpk',lpopts={},witness=None,xN=None,face=None,adj=None):
        # INTrep = INTrep[0]
        # We assume INTrep is a list of integers representing the hyperplanes that CAN'T be flipped
        # t = time.time()
        witnessList = []
        INTrep, boolIdxNoFlip, intIdx, intIdxNoFlip = self.decodeRegionStore(INTrep)


        # Flip the un-flippable hyperplanes; this must be undone later
        H[INTrep,:] = -H[INTrep,:]


        d = H.shape[1]-1



        constraint_list = None


        faces, witnessList = self.thisProxy[self.thisIndex].concreteMinHRep(H,constraint_list,boolIdxNoFlip,intIdxNoFlip,intIdx,solver=solver,interiorPoint=witness,ret=True).get()

        successors = []
        for idx in range(len(faces)):
            i = faces[idx]
            if boolIdxNoFlip[intIdx[i]//8] & 1<<(intIdx[i] % 8) == 0:
                # boolIdxNoFlip[intIdx[i]] = 1
                # t = time.time()
                boolIdxNoFlip[intIdx[i]//8] = boolIdxNoFlip[intIdx[i]//8] | 1<<(intIdx[i] % 8)
                insertIdx = 0
                while insertIdx < len(intIdxNoFlip) and intIdxNoFlip[insertIdx] < intIdx[i]:
                    insertIdx += 1
                temp = copy(intIdxNoFlip)
                temp.insert(insertIdx,intIdx[i])
                successors.append( \
                        [ copy(boolIdxNoFlip), tuple(temp), self.flippedConstraints.N, (intIdx[i],) if self.sendFaces else tuple() , None if witness is None else witnessList[idx], None, None ]
                    )
                boolIdxNoFlip[intIdx[i]//8] = boolIdxNoFlip[intIdx[i]//8] ^ 1<<(intIdx[i] % 8)
                # self.conversionTime += time.time() - t
                t = time.time()
                if not self.doRS:
                    cont = self.thisProxy[self.thisIndex].hashAndSend(successors[-1],ret=True).get()
                else:
                    cont = True
                self.stats['xferTime'] += time.time() - t

                if not cont:
                    H[INTrep,:] = -H[INTrep,:]
                    return successors, -1, witnessList
        if not self.doRS and self.sendFaces and not isinstance(self.clearTable,str):
            self.thisProxy[self.thisIndex].hashAndSend([copy(boolIdxNoFlip),copy(intIdxNoFlip),self.flippedConstraints.N,[copy(ii[3][0]) for ii in successors]], ret=True).get()
        # facesInt = np.full(self.N,0,dtype=bool)
        sel = tuple(np.array(intIdx,dtype=np.uint64)[faces].tolist())
        # facesInt[sel] = np.full(len(sel),1,dtype=bool)

        # Undo the flip we did before, since it affects a referenced (as opposed to copied) array:
        H[INTrep,:] = -H[INTrep,:]

        # return [successors, bytes(np.packbits(facesInt,bitorder='little'))]
        if not self.doRS:
            return [], sel, witnessList
        else:
            return successors, sel, witnessList

    @coro
    def processNodeSuccessorsInsertHyperplane(self,INTrep,N,H,payload=[],solver='glpk',lpopts={},witness=None,xN=None,face=None,adj=None):
        # *****
        # This function takes as input the regions newly split by the inserted hyperplane.
        #
        # It then has _____ main functions:
        #   1) It should compute the successors of the split region (in the form of newly split regions
        #      from the successor of the currently split region)
        #   2) These new successor should be seeded with the correct face information, computed by
        #      querying the poset
        #   3) The hashAndSend calls should correctly update the adjacency encoding information for the
        #      regions in the new level, as stored in the adj property of the node
        #
        # This function takes as input one of the regions on the current level that is to be split
        # in the form of a canonical choice of one of its new subregions specified by N hyperplanes.
        # This region will contain in its face attribute ALL of the faces of the region to be replaced!
        #
        # It then has four tasks:
        #     1) edit the face information of this canoncial region to reflect the split
        #     2) create the other split subregion (if required), with correct face information from 1)
        #     3) delete the old region that these (two) split region(s) replace
        #     4) Hash a canonical region (in N hyperplanes) for each of the replaced region's successors
        # *****
        debug = False
        d = H.shape[1]-1
        witnessList = []
        Ntab = N

        # Note the N passed here includes the inserted hyperplane, and INTrep will always be of the same length
        if debug:
            print(f'///// extrasINTrep0 {charm.myPe()}  INTrep = {INTrep}; {adj}')
        boolIdxNoFlip, INTrep, _ = region_helpers.recodeRegNewN(-N + Ntab, INTrep , N) # should be identity for INTrep
        # Wrong! we should only ever process nodes encoded using N... That is INTrep == INTrepFull should ALWAYS HOLD
        # extrasINTrep = tuple( np.nonzero( -H[Ntab:(N-1),1:] @ witness >= (1 + self.rTol) * H[Ntab:(N-1),0].reshape(-1,1) + self.tol )[0] + Ntab )
        extrasINTrep = tuple()
        if debug:
            print(f'///// extrasINTrep1 {charm.myPe()} = {extrasINTrep}; INTrep = {INTrep}')
        boolIdxNoFlipFull, INTrepFull, _ = region_helpers.recodeRegNewN(0, INTrep + extrasINTrep , N)
        if debug:
            print(f'///// extrasINTrep2 {charm.myPe()} = {extrasINTrep}; INTrepFull = {INTrepFull} boolIdxNoFlip = {boolIdxNoFlipFull}')
            print(f'///// extrasINTrep3 {charm.myPe()} = {extrasINTrep}; INTrep = {INTrep} boolIdxNoFlip = {boolIdxNoFlip}')
        # INTrep, boolIdxNoFlip, intIdx, intIdxNoFlip = self.decodeRegionStore(INTrep)

        rebasedINTrep = region_helpers.recodeRegNewN(N-Ntab,self.flippedConstraints.rebaseRegion(INTrepFull)[0],N)[1]
        rebasedINTrepSet = set(rebasedINTrep)
        if debug:
            print(f'///// rebasedINTrep = {rebasedINTrep}')

        initialRegion = True


        # Ignore negative-side inserted regions (inserted hyperplane cannot have a full-dimensional
        # intersection with an existsing hyperplane -- handled by vectorSet uniqueness)
        if len(INTrepFull) > 0 and INTrepFull[-1] == N-1:
            if debug:
                print(f'----[[[{INTrepFull}, {charm.myPe()}]]]    Ignoring negative-side region')
            return [set([]),None]

        assert len(adj) == 1, f'Incorrect incomming adj parameter'
        if debug:
            print(f'adj = {adj}')
            print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    submittedDeleteQuery = {-N + adj[-1]} {region_helpers.recodeRegNewN(-N + adj[-1], INTrep, N)}')
        q = self.thisProxy[self.thisIndex].query( \
                                    region_helpers.recodeRegNewN( \
                                        -N + adj[-1], \
                                        INTrep, \
                                        N \
                                    ), \
                                    op=DistributedHash.QUERYOP_DELETE, ret=True)
        # Get results of query, and use it to set adj correctly
        q = q.get()
        if q[0] > 0:
            oldFace = q[1]
            oldWitness = q[2]
            adj = q[3]
            oldPayload = q[4]
        else:
            adj = None
            if debug:
                print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    WARNING {charm.myPe()}: Unable to find successor region to delete! {neighborReg} {adj}')
                print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    query results = {q}; adjUpdate = {adj}')
        adj[-1] = N
        if debug:
            print(f' %%%%%%%% MADE IT HERE! %%%%%%%%')
            print(f'........... N={N} {adj}')

        Ntab = adj[-1]
        incommingFace = face
        allFace = set(adj.keys())
        allFace.remove(-1)
        face = allFace - incommingFace
        if debug:
            print(f'........... allFace = {allFace}; incommingFace = {incommingFace}; incomming adj = {adj}')

        validFlips = face - rebasedINTrepSet
        validFlipsList = sorted(list(validFlips))

        # Find all of the adjacent regions that touch the inserted hyperplane
        splitRegions = []

        if debug:
            print(f'----[[[{INTrepFull}, {charm.myPe()}]]]   INTrepSetFull = {INTrepFull}')
            print(f'----[[[{INTrepFull}, {charm.myPe()}]]]   Valid flips = {validFlips}')
            print(f'----[[[{INTrepFull}, {charm.myPe()}]]]   rebasePt = {self.flippedConstraints.rebasePt.flatten()}')
            print(f'----[[[{INTrepFull}, {charm.myPe()}]]]   ptLift   = {self.iPtLift.flatten()}')

        # The rebasedINTrep should be -- by construction -- an 'expanded' region specification
        # for the projected constraints.
        # Hence, collapseRegion should give us a valid region specification in the projected
        # hyperplane arrangement, ignoring degenerate *faces* in that arrangement. Thus, we
        # can use lpMinHRep, combined with the expand duplicates method of vectorSet (talk
        # about creating the right abstractions! 😄), to identify all full-dimensional faces of
        # this region that are split by the inserted hyperplane.
        projINTrep = self.iFlipConstraints.collapseRegion(rebasedINTrep)
        if debug:
            print(f'----[[[{INTrepFull}, {charm.myPe()}]]]   temp = {projINTrep}; rebasedINTrepSet = {rebasedINTrep}')

        # Now let's find which of the validFlips (unflipped hyperplanes starting from the
        # region containing rebasePt of self.flippedConstraints) correspond to faces in the
        # projected hyperplane arrangement -- these will be the full dimensional faces of
        # the current region that are split by the inserted hyperplane
        validFlipsListExpand = list(itertools.chain.from_iterable([self.iFlipConstraints.hyperSet.expandDuplicates(i) for i in validFlipsList]))
        collapsedFaces = self.iFlipConstraints.collapseRegion( validFlipsListExpand )
        if debug:
            print(f'++++[[[{INTrepFull}, {charm.myPe()}]]]   validFlipsList = {validFlipsList}; validFlipsListExpand = {validFlipsListExpand}; collapseRegion = {self.iFlipConstraints.collapseRegion( validFlipsList )} {self.iFlipConstraints.nonRedundantHyperplanes}')
            print(f'++++[[[{INTrepFull}, {charm.myPe()}]]]   collapsedFaces = {collapsedFaces}')
        splitFacesIdx, _ = self.thisProxy[self.thisIndex].concreteMinHRep( \
                                            self.iFlipConstraints.getRegionConstraints( projINTrep, allN=False  ), \
                                            None, \
                                            # These arguments are only used with self.useQuery=True, which is disabled
                                            # automatically for insertHyperplane mode
                                            bytearray(b''), \
                                            tuple(), \
                                            # This is the only argument that is relevant for the minHRep
                                            collapsedFaces, \
                                            solver = self.solver, \
                                            ret = True \
                                        ).get()
        if debug:
            print(f'++++[[[{INTrepFull}, {charm.myPe()}]]]   {[(iexp:=self.iFlipConstraints.nonRedundantHyperplanes[collapsedFaces[i]],self.iFlipConstraints.hyperSet.expandDuplicates(iexp)) for i in splitFacesIdx]}')
        splitFaces = {(self.iFlipConstraints.nonRedundantHyperplanes[(iexp:=collapsedFaces[i])]): \
                      self.iFlipConstraints.hyperSet.expandDuplicates(iexp) for i in splitFacesIdx}
                      # [validFlipsList[ii] for ii in self.iFlipConstraints.hyperSet.expandDuplicates(iexp)] for i in splitFacesIdx}
        if debug:
            print(f'----[[[{INTrepFull}, {charm.myPe()}]]]    splitFaces = {splitFaces}')
        allSplitFaces = set()
        for h, dups in splitFaces.items():
            allSplitFaces |= set(dups)
        if debug:
            print(f'----[[[{INTrepFull}, {charm.myPe()}]]]    allSplitFaces = {allSplitFaces}')
        nonSplitFaces = validFlips - allSplitFaces
        if debug:
            print(f'----[[[{INTrepFull}, {charm.myPe()}]]]    nonSplitFaces = {nonSplitFaces}')


        # Ok.... So how do we change this to make it work properly?

        # 1) Each split face will correspond to two new regions, so we need to
        #   a) Delete the full-dimensional region corresponding to this face
        #       (if it is duplicated, then flip ALL of the duplicates to get the region)
        #   b) Replace it with a recoded region using N hyperplanes, and seed it with a hashAndSend call
        # 2) Replace the adj value for each abutting region to reflect that the current region (and
        # the to-be-added negative-side region) are encoded using N hyperplanes. This will entail many
        # redundant hashAndSend calls (i.e. the same regions will be hit more than once)
        # 3) Calculate the correct faces for the newly inserted regions
        #   a) Use 2) to update the replaced region in 1)b). This is necessary because the correct
        #       face information isn't yet computed when the new node is created in 1b)

        for h in allFace - {N-1}:
            INTrepSetFull = set(INTrepFull)
            if debug:
                print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    Working on face {h}')
                print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    INTrepSetFull = {INTrepSetFull}')
            neighborReg = sorted(list(INTrepSetFull | {h} if not h in INTrepSetFull else INTrepSetFull - {h}))
            if debug:
                print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    neighborReg = {neighborReg} {adj[h]}')
            cont = self.thisProxy[self.thisIndex].hashAndSend( \
                        region_helpers.recodeRegNewN( \
                            -N + (adj[h]), \
                            neighborReg, \
                            N \
                        ) + ( \
                            set(), \
                            witness \
                        ), \
                        adjUpdate = {h:N}, \
                        ret=True \
                    ).get()
            testStripNum = -N + adj[h]
            if debug:
                print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    submitted update = {testStripNum} {region_helpers.recodeRegNewN(testStripNum, neighborReg, N)}')
        for h in splitFaces:
            neighborReg = copy(INTrepSetFull)
            for hh in splitFaces[h]:
                if not hh in neighborReg:
                    neighborReg |= {hh}
                else:
                    neighborReg = neighborReg - {hh}
            neighborReg = sorted(list(neighborReg))
            if debug:
                print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    neighborReg = {neighborReg}')
            if len(splitFaces[h]) == 1:
                stripNum = N - adj[h]
            else:
                # Iteratively query until we find the right encoding for the target region
                stripNum = None
                for idx in range(1,self.flippedConstraints.N - self.flippedConstraints.baseN + 1):
                    newBaseReg, newBaseRegTup, newBaseRegN = region_helpers.recodeRegNewN(-idx, neighborReg, self.flippedConstraints.N)
                    print((newBaseReg, newBaseRegTup, newBaseRegN))
                    retVal = self.thisProxy[self.thisIndex].query([newBaseReg, newBaseRegTup, newBaseRegN],ret=True).get()
                    print(retVal)
                    if retVal[0] > 0:
                        stripNum = idx
                        break
                if stripNum is None:
                    print(f'ERROR finding desired region!')
                    return [set([]),None]
            if debug:
                print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    stripNum = {stripNum}')

            H[neighborReg,:] = -H[neighborReg,:]
            intPtPos = region_helpers.findInteriorPoint( \
                                H, \
                                solver=self.solver, \
                                lpObj=self.lp, \
                                tol=self.tol, \
                                rTol=self.rTol, \
                                lpopts=lpopts \
                            )
            if debug:
                print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    intPtPos = {intPtPos}')
            H[neighborReg,:] = -H[neighborReg,:]
            # Create the new split region (positive-side of inserted hyperplane)
            cont = self.thisProxy[self.thisIndex].hashAndSend( \
                                region_helpers.recodeRegNewN( \
                                    0, \
                                    neighborReg, \
                                    N \
                                ) + ( \
                                    set(splitFaces[h]) if len(splitFaces[h]) == 1 else set(), \
                                    intPtPos \
                                ), \
                                adjUpdate={-1:N-stripNum}, \
                                payload=None, \
                                ret=True \
                            ).get()


        # Now fix the face information for the current node

        # calculate faces for new region on positive side of inserted hyperplane
        H[INTrepFull,:] = -H[INTrepFull,:]
        posFaces = set()
        splitFacesUnique = set( itertools.chain.from_iterable([splitFaces[h] for h in splitFaces if len(splitFaces[h]) == 1]) ) & allFace
        # splitFacesRedundant = set( itertools.chain.from_iterable([splitFaces[h] for h in splitFaces if len(splitFaces[h]) != 1]) ) & allFace
        if debug:
            print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    splitFacesUnique = {splitFacesUnique}')
            # print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    splitFacesRedundant = {splitFacesRedundant}')
            print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    face = {face}')
            print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    witness = {witness}')
        splitFacesSet = incommingFace | splitFacesUnique
        sel = sorted([ f for f in face if not f in splitFacesSet ])
        if debug:
            print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    sel = {sel}')
        posFacesIdx, _ = self.thisProxy[self.thisIndex].concreteMinHRep( \
                                            H, \
                                            None, \
                                            # These arguments are only used with self.useQuery=True, which is disabled
                                            # automatically for insertHyperplane mode
                                            bytearray(b''), \
                                            tuple(), \
                                            # This is the only argument that is relevant for the minHRep
                                            sel, \
                                            solver = self.solver, \
                                            ret = True \
                                        ).get()
        posFaces = set([ sel[f] for f in posFacesIdx ])
        posFaces.add(N-1)
        if debug:
            print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    posFaces = {posFaces}')
        H[INTrepFull,:] = -H[INTrepFull,:]

        # NB: splitFacesSet contains ALL faces that are for sure split into full-dimensional faces by the
        # inserted hyperplane, i.e. are faces to BOTH the positive and negative inserted region
        posFaces |= splitFacesSet
        negFaces = (allFace - posFaces) | splitFacesSet
        negFaces.add(N-1)
        posFacesUpdate = {i:(adj[i] if i in posFaces else None) for i in allFace}
        for i in splitFacesSet:
            posFacesUpdate[i] = N
        posFacesUpdate[N-1] = N
        posFacesUpdate[-1] = N
        if debug:
            print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    posFaces = {posFaces}')
            print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    negFaces = {negFaces}')
            print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    splitFacesSet = {splitFacesSet}')
            print(f'    ----[[[{INTrepFull}, {charm.myPe()}]]]    posFacesUpdate = {posFacesUpdate}')

        # Send an update to the CURRENT region, using read/write semantics of adj dictionary
        cont = self.thisProxy[self.thisIndex].hashAndSend( \
                                    region_helpers.recodeRegNewN( \
                                        0, \
                                        INTrepFull, \
                                        N \
                                    ) + ( \
                                        posFaces, \
                                        witness \
                                    ), \
                                    adjUpdate=posFacesUpdate, \
                                    payload=payload, \
                                    ret=True \
                                ).get()

        # We are for sure working on a replaced node, encoded using all N hyperplanes (including insertion)
        # that is also on the positive side of the inserted hyperplane.
        # Thus, we insert its companion, negative version:
        H[INTrepFull,:] = -H[INTrepFull,:]
        H[N-1,:] = -H[N-1,:]
        intPtNeg = region_helpers.findInteriorPoint( \
                                    H, \
                                    solver=self.solver, \
                                    lpObj=self.lp, \
                                    tol=self.tol, \
                                    rTol=self.rTol, \
                                    lpopts=lpopts \
                                )
        H[N-1,:] = -H[N-1,:]
        H[INTrepFull,:] = -H[INTrepFull,:]
        negAdj = {i:(adj[i] if i in adj else N) for i in negFaces}
        for i in splitFacesSet:
            negAdj[i] = N
        negAdj[-1] = N
        if debug:
            print(f'----[[[{INTrepFull}, {charm.myPe()}]]]    Sending negative side region intPtNeg = {intPtNeg}; {self.flippedConstraints.N} {negAdj}')
        cont = self.thisProxy[self.thisIndex].hashAndSend( \
                                    region_helpers.recodeRegNewN( \
                                        0, \
                                        INTrepFull + (N-1,), \
                                        N \
                                    ) + ( \
                                        negFaces, \
                                        intPtNeg \
                                    ), \
                                    adjUpdate=negAdj, \
                                    payload=payload, \
                                    ret=True \
                                ).get()
        if debug:
            print(f'----[[[{INTrepFull}, {charm.myPe()}]]]    DONE sending negative side region intPtNeg = {intPtNeg}; {self.flippedConstraints.N}')

        constraint_list = None


        return [set([]),None]

    @coro
    def processNodeSuccessorsCanonicalizeTable(self,INTrep,N,H,payload=[],solver='glpk',lpopts={},witness=None,xN=None,face=None,adj=None):
        debug = False
        d = H.shape[1]-1
        witnessList = []

        if not self.iPtLift is None and isinstance(self.iPtLift,np.ndarray) and self.iPtLift.shape == (d,1):
            self.flippedConstraints.setRebase(self.iPtLift)
            self.iPtLift = None
            self.useRebase = True

        boolIdxNoFlip, INTrep, _ = region_helpers.recodeRegNewN(0, INTrep , N) # should be identity for INTrep

        if witness is None:
            idx = np.zeros(H.shape[0], dtype=np.bool_)
            idx[self.flippedConstraints.N:] = np.ones_like(idx[self.flippedConstraints.N:],dtype=np.bool_)
            idx[:N] = np.ones_like(idx[:N],dtype=np.bool_)

            Hlocal = H[idx,:]
            Hlocal[INTrep,:] = -Hlocal[INTrep,:]

            if not face is None:
                Hlocal = np.vstack([Hlocal[sorted(list(face)),:], Hlocal[N:,:]])

            witness = region_helpers.findInteriorPoint( \
                            Hlocal, \
                            solver=self.solver, \
                            lpObj=self.lp, \
                            tol=self.tol, \
                            rTol=self.rTol, \
                            lpopts=lpopts \
                        )
            if witness is None:
                print(f'ERROR finding interior point!')
                return [set([]),None]

        # This is a region encoded with fewer than the final number of hyperplanes, self.flippedConstraints.N,
        # so we have to figure out which side of the un-encoded hyperplanes this region lies on
        if self.flippedConstraints.N - N > 0:
            newBaseRegFullTup = INTrep + tuple( \
                                N + np.nonzero( \
                                    ( \
                                        -H[N:self.flippedConstraints.N,1:] @ witness \
                                        - H[N:self.flippedConstraints.N,0].reshape(-1,1) \
                                    ).flatten() \
                                    >= self.tol \
                                )[0] \
                            )
        else:
            newBaseRegFullTup = INTrep

        if self.useRebase:
            _, newBaseRegFullTup = self.flippedConstraints.rebaseRegion(newBaseRegFullTup)

        cont = self.thisProxy[self.thisIndex].hashAndSend( \
                                    region_helpers.recodeRegNewN( \
                                        0, \
                                        newBaseRegFullTup, \
                                        self.flippedConstraints.N \
                                    ) + ( \
                                        face, \
                                        witness \
                                    ), \
                                    adjUpdate={ky:self.flippedConstraints.N for ky in adj.keys()} if isinstance(adj,dict) else None, \
                                    payload=payload, \
                                    ret=True \
                                ).get()

        return [set([]),None]



def Union(contribs):
    return set().union(*contribs)

Reducer.addReducer(Union)

def DictAccum(contribs):
    result = defaultdict(int)
    for trib in contribs:
        for ky in trib.keys():
            result[ky] += trib[ky]
    return result

Reducer.addReducer(DictAccum)

def DictListJoin(contribs):
    result = defaultdict(list)
    for trib in contribs:
        for ky in trib.keys():
            result[ky] += trib[ky]
    return result

Reducer.addReducer(DictListJoin)

def DictSetUnion(contribs):
    result = defaultdict(set)
    for trib in contribs:
        for ky in trib.keys():
            result[ky] |= trib[ky]
    return result

Reducer.addReducer(DictSetUnion)

