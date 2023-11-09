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
        if adj and isinstance(adj,dict):
            for ky in adj.keys():
                self.adj[ky] = adj[ky]
        # self.update(lsb, msb, nodeBytes, N, originPe, face, witness, adj, *args)


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
        self.N = len(self.AbPairs[0][0])


    @coro
    def setConstraint(self,lb=0,out=0,timeout=None,prefilter=True,rebasePt=None):
        self.populated = False
        self.incomplete = True
        self.N = len(self.AbPairs[0][0])
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
        defaultSettings = ['clearTable','retrieveFaces','verbose','sendFaces','queryReturnInfo']
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
        self.successorProxies[0].hashAndSend([ \
                                        boolIdxNoFlip, \
                                        thisLevel[0][0], \
                                        self.flippedConstraints.N, \
                                        tuple() if face is None else face, \
                                        self.flippedConstraints.pt if witness is None else witness \
                                    ], \
                                    adjUpdate=(False if not adjUpdate else adjUpdate), \
                                    payload=(tuple() if payload is None else payload), \
                                    vertex=(None if self.hashStoreMode != 2 else (self.flippedConstraints.pt,tuple())), \
                                ret=True).get()
        thisLevel = [( \
                      boolIdxNoFlip if self.hashStoreMode == 1 else thisLevel[0][0], \
                      self.flippedConstraints.N, \
                      0, \
                      tuple() if face is None else face, \
                      self.flippedConstraints.pt if witness is None else witness, \
                      False if not adjUpdate else adjUpdate, \
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
        while level < self.N+1 and nextLevelSize > 0:
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
        # To do?: have each instance insert the hyperplane separately for speed (avoids copying custom object)
        aug = self.flippedConstraints.deserialize()
        if not inserted:
            self.flippedConstraints.serialize()
            self.succGroup.initialize(aug.N, aug, None, awaitable=True).get()
            self.localVarGroup.setConstraintsOnly(aug,awaitable=True).get()
            return True

        localOpts = deepcopy(opts)
        localOpts['method'] = 'insertHyperplane'
        localOpts['clearTable'] = False
        localOpts['sendWitness'] = True
        localOpts['sendFaces'] = True
        localOpts['queryReturnInfo'] = True
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

        print(np.abs(-hyper[1:].reshape(1,-1) @ ptLift - hyper[0]))

        newBaseRegFullTup = tuple(np.nonzero((-aug.constraints[:(aug.N-1),1:] @ ptLift - aug.constraints[:(aug.N-1),0].reshape(-1,1)).flatten() >= tol)[0])
        newBaseRegFull = region_helpers.tupToBytes(newBaseRegFullTup, *region_helpers.byteLenFromN(aug.N))
        rebasePt = region_helpers.findInteriorPoint(aug.getRegionConstraints(newBaseRegFullTup),solver=solver,tol=tol,rTol=rTol,lpopts=lpopts)
        if rebasePt is None:
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
            newBaseReg, newBaseRegTup, newBaseRegN = region_helpers.recodeRegNewN(-idx, newBaseRegFull, aug.N)
            print((newBaseReg, newBaseRegTup, newBaseRegN))
            retVal = self.succGroup[0].query([newBaseReg, newBaseRegTup, newBaseRegN],ret=True).get()
            print(retVal)
            if retVal[0] > 0:
                stripNum = idx
                break

        self.distHashTable.awaitPending(usePosetChecking=False,awaitable=True).get()
        self.succGroup.sendAll(-2,awaitable=True).get()
        self.succGroup.closeQueryChannels(awaitable=True).get()
        self.succGroup.flushMessages(ret=True).get()

        aug.root = tuple(newBaseRegFullTup)
        aug.setRebase(rebasePt)
        self.succGroup.initialize(aug.N, aug.serialize(), None, awaitable=True).get()
        self.localVarGroup.setConstraintsOnly(aug.serialize(),awaitable=True).get()

        self.distHashTable.setCheckDispatch({'check':'checkForInsert','update':'updateForInsert'},awaitable=True).get()

        newAdj = deepcopy(retVal[3])
        newAdj[-1] = aug.N - stripNum
        for f in retVal[1]:
            if not f in newAdj:
                newAdj[f] = self.flippedConstraints.baseN
        print(newAdj)

        self.populated = False
        self.thisProxy.populatePoset(face=copy(retVal[1]),witness=rebasePt,adjUpdate=newAdj,payload=deepcopy(retVal[4]),opts=localOpts,awaitable=True).get()

        # Now that we're all done, restore default dispatch for check/update
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
        self.rsScheduler = rsScheduler
        self.rsPeFree = True
        self.rsDone = False
        self.rsDepth = 0
        self.rsRegionCount = 0
        self.sendFaces = False
        self.sendWitness = False
        self.deferLock = False
        self.hashedNodeCount = 0

    def initialize(self,N,constraints,timeout):
        self.workInts = []
        self.N = N
        self.flippedConstraints = constraints.deserialize()
        self.constraints = self.flippedConstraints.constraints
        # self.processNodeSuccessors = partial(successorWorker.processNodeSuccessorsFastLP, self, solver='glpk')
        self.processNodeSuccessors = self.thisProxy[self.thisIndex].processNodeSuccessorsFastLP
        self.processNodesArgs = {'solver':'glpk','ret':True}
        # Defaults to glpk, so this empty call is ok:
        self.lp = encapsulateLP.encapsulateLP()
        self.lpIntPoint = encapsulateLP.encapsulateLP()
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
    @coro
    def getTimeout(self):
        return self.timedOut

    def setMethod(self,method='fastLP',solver='glpk',useQuery=False,lpopts={},reverseSearch=False,hashStore='bits',tol=1e-9,rTol=1e-9,sendFaces=False,clearTable='speed',sendWitness=False,verbose=True):
        self.lp.initSolver(solver=solver, opts={'dim':(self.constraints.shape[1]-1)})
        self.lpIntPoint.initSolver(solver=solver, opts={'dim':(self.constraints.shape[1])})
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
            self.stats['LPSolverCount'] = self.lp.lpCount + self.lpIntPoint.lpCount
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

    def hashNode(self,toHash,payload=None,vertex=None,adjUpdate=False):
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
    def hashAndSend(self,toHash,payload=None,vertex=None,adjUpdate=False):
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
    def query(self, q, op=None):
        qOp = 0
        if isinstance(op,int):
            qOp = op
        # print('PE' + str(charm.myPe()) + ' Query to send is ' + str(q))
        self.stats['numQueries'] += 1
        val = self.hashNode(q)
        self.queryChannels[val[0]].send((qOp,) + val)
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
                successorList[ii] = self.processNodeSuccessors(self.workInts[ii][0],self.N,self.constraints,**self.processNodesArgs,witness=self.workInts[ii][4], payload=self.workInts[ii][6],xN=self.workInts[ii][1],face=self.workInts[ii][3],adj=self.workInts[ii][5], awaitable=True).get()
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
                status, x = self.lp.runLP( \
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
                x = region_helpers.findInteriorPoint(H,solver=solver,lpObj=self.lpIntPoint,tol=self.tol,rTol=self.rTol,lpopts=self.lpopts)
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
        d = H.shape[1]-1
        witnessList = []
        Ntab = adj[-1]
        # Note the N passed here includes the inserted hyperplane, and INTrep will always be of the same length
        boolIdxNoFlipFull, INTrepFull, _ = region_helpers.recodeRegNewN(0, INTrep, N)
        boolIdxNoFlip, INTrep, _ = region_helpers.recodeRegNewN(N - Ntab, INTrep, N)
        # INTrep, boolIdxNoFlip, intIdx, intIdxNoFlip = self.decodeRegionStore(INTrep)

        rebasedINTrep = region_helpers.recodeRegNewN(N-Ntab,self.flippedConstraints.rebaseRegion(INTrepFull)[0],N)[1]
        rebasedINTrepSet = set(rebasedINTrep)
        print(f'///// rebasedINTrep = {rebasedINTrep}')

        # Note for clarity: the if condition is only needed on the first pass, since all subsequently processed nodes
        # will be encoded using N hyperplanes (old nodes will be deleted and re-inserted)
        if len(rebasedINTrep) == 0 and Ntab < N:
            initialRegion = True
            newBaseRegFullTup = tuple( \
                                    np.nonzero( \
                                        ( \
                                            -self.flippedConstraints.constraints[:(self.flippedConstraints.N-1),1:] @ witness \
                                            - self.flippedConstraints.constraints[:(self.flippedConstraints.N-1),0].reshape(-1,1) \
                                        ).flatten() \
                                        >= self.tol \
                                    )[0] \
                                )
            boolIdxNoFlipFull, INTrepFull, _ = region_helpers.recodeRegNewN(0, newBaseRegFullTup, N)
            boolIdxNoFlip, INTrep, _ = region_helpers.recodeRegNewN(N - Ntab, INTrepFull, N)
            rebasedINTrep = region_helpers.recodeRegNewN(N-Ntab,self.flippedConstraints.rebaseRegion(INTrepFull)[0],N)[1]
            rebasedINTrepSet = set(rebasedINTrep)
        else:
            initialRegion = False

        validFlips = face - rebasedINTrepSet

        # We have to retrieve the information from the node in the table that is going to be split
        q = self.thisProxy[self.thisIndex].query( [boolIdxNoFlip, INTrep, Ntab], op=DistributedHash.QUERYOP_DELETE, ret=True).get()
        if q[0] > 0:
            oldFace = q[1]
            oldWitness = q[2]
            oldAdj = q[3]
            oldPayload = q[4]
        print(f'()()()    q = {q}')


        # Flip the un-flippable hyperplanes; this must be undone later
        # Again note that H contains the inserted hyperplane
        H[INTrepFull,:] = -H[INTrepFull,:]


        # Find all of the adjacent regions that touch the inserted hyperplane
        splitRegions = []


        for hh in validFlips:
            faceWitnesses = None
            INTrepSet = set(INTrepFull)
            if initialRegion:
                oldFace = face
                oldWitness = witness
                oldAdj = adj
                oldPayload = payload
                h = -1
                print(f'    """""""" Setting initial region properties oldFace = {oldFace}; oldAdj = {oldAdj}')
            else:
                h = hh
                H[h,:] = -H[h,:]
                INTrepSet = INTrepSet | {h} if not h in INTrepSet else INTrepSet - {h}
                q = self.thisProxy[self.thisIndex].query( region_helpers.recodeRegNewN(N-adj[h], sorted(tuple(INTrepSet)), N) , ret=True).get()
                if q[0] > 0:
                    oldFace = q[1]
                    oldWitness = q[2]
                    oldAdj = q[3]
                    oldPayload = q[4]
            for sgn in [1, -1]:
                H[N-1,:] = sgn * H[N-1,:]
                intPt = region_helpers.findInteriorPoint( \
                                            H, \
                                            solver=self.solver, \
                                            lpObj=self.lp, \
                                            tol=self.tol, \
                                            rTol=self.rTol, \
                                            lpopts=lpopts \
                                        )
                print(f'    """""""" Received interior point {intPt}')
                H[N-1,:] = sgn * H[N-1,:]
                if not intPt is None:
                    # We have a full dimensional region that is split by the inserted hyperplane, so hash it
                    # We have to calculate the new face information first
                    if faceWitnesses is None:
                        faceWitnesses = {}
                        holder = {}
                        qs = []
                        for f in (oldFace & validFlips) - {h}:
                            if f == N-1: continue
                            s = sorted(tuple(INTrepSet | {f})) if not f in INTrepSet else sorted(tuple(INTrepSet - {f}))
                            print(f'    """""""" computed adjacency region {s}')
                            faceWitnesses[f] = self.thisProxy[self.thisIndex].query(region_helpers.recodeRegNewN(-N + oldAdj[f], s, N ),ret=True).get()
                    # faceWitnesses[h] = intPt
                    print(f'    """""""" faceWitnesses = {faceWitnesses}')
                    newFaces = [N-1]
                    for f in faceWitnesses.keys():
                        # Now check the witness for f to see which side of the inserted hyperplane it's on
                        wit = sgn * faceWitnesses[f][2]
                        if -H[N-1,1:].reshape(-1,1) @ wit >= sgn * H[N-1,0] + self.tol:
                            # Face witness is on the same side of inserted hyperplane as intPt
                            newFaces.append(f)

                    INTrepSet = set(INTrepFull)
                    if sgn < 0:
                        s = sorted(INTrepFull + tuple(N-1) + (tuple(h) if not initialRegion else tuple()))
                    else:
                        s = sorted(INTrepFull + (tuple(h) if not initialRegion else tuple()))
                    oldAdj[h] = N
                    cont = self.thisProxy[self.thisIndex].hashAndSend( \
                                                        region_helpers.recodeRegNewN( \
                                                            0, \
                                                            s, \
                                                            N \
                                                         ) + ( \
                                                            newFaces, \
                                                            intPt, \
                                                            oldAdj, \
                                                            oldPayload \
                                                         ), \
                                                        ret=True \
                                                    ).get()
            if not initialRegion:
                H[h,:] = -H[h,:]
            faceWitnesses = None

        constraint_list = None


        H[INTrepFull,:] = -H[INTrepFull,:]




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

