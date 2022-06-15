from platform import node
from typing import Dict
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
# import TLLHypercubeReach 
import posetFastCharm_numba
import region_helpers

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
        self.constraints = self.localProxy[self.storePe].getConstraints(ret=True).get()

    # def update(self):
    #     pass

    # def check(self):
    #     pass

class localVar(Chare):
    def setConstraints(self,constraints):
        self.constraints = constraints
        self.schedCount = 0
        self.skip = False
    def getConstraints(self):
        return self.constraints
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

class Poset(Chare):
    
    @coro
    def init(self, peSpec, nodeConstructor, localVarGroup, successorChare):
        
        # self.stackNum = batchSize
        # To do: check to make sure we're passed a valid Group in localVarGroup
        self.localVarGroup = localVarGroup
        self.useDefaultLocalVarGroup = False
        if localVarGroup is None:
            self.useDefaultLocalVarGroup = True
            self.localVarGroup = Group(localVar,args=[])
            charm.awaitCreation(self.localVarGroup)
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
        self.succGroupFull.initPEs(self.posetPElist)
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
            self.posetPEs \
        ],onPE=self.hashPElist[0])
        charm.awaitCreation(self.distHashTable)
        # print('Initialized distHashTable group')
        initFut = self.distHashTable.initialize(awaitable=True)
        initFut.get()

    
    def initialize(self, AbPairs, pt, fixedA, fixedb, normalize=1.0):
        self.AbPairs = deepcopy(AbPairs)
        self.pt = pt
        self.fixedA = fixedA.copy()
        self.fixedb = fixedb.copy()
        self.normalize = normalize

        if normalize > 0:
            for out in range(len(AbPairs)):
                self.nrms = self.normalize / np.linalg.norm(self.AbPairs[out][0],axis=1).reshape(-1,1)
                self.AbPairs[out][0] = self.nrms * self.AbPairs[out][0]
                self.AbPairs[out][1] = self.nrms * self.AbPairs[out][1]
            nrms = np.linalg.norm(self.fixedA,axis=1).reshape(-1,1) * self.normalize
            self.fixedA = (self.normalize/nrms) * self.fixedA
            self.fixedb = (self.normalize/nrms) * self.fixedb
        else:
            self.nrms = np.ones((self.N,1))


        # self.N = len(self.AbPairs[0][0])
        # self.wholeBytes = (self.N + 7) // 8
        # self.tailBits = self.N - 8*(self.N // 8)


    @coro
    def setConstraint(self,lb=0,out=0,timeout=None,prefilter=True):
        self.populated = False
        self.incomplete = True
        self.N = len(self.AbPairs[0][0])
        if prefilter:
            createConstraints = region_helpers.flipConstraintsReducedMin
        else:
            createConstraints = region_helpers.flipConstraints
        self.flippedConstraints = createConstraints( \
                -1*self.AbPairs[out][0], \
                self.AbPairs[out][1] - lb*self.nrms, \
                self.pt, \
                self.fixedA, \
                self.fixedb \
            )
        self.N = self.flippedConstraints.N
        
        
        stat = self.succGroup.initialize(self.N,self.flippedConstraints,timeout,awaitable=True)
        stat.get()
        if self.useDefaultLocalVarGroup:
            self.localVarGroup.setConstraints(self.flippedConstraints,awaitable=True).get()

        self.populated = False

        return 1

    @coro
    def getConstraintsObject(self):
        return self.flippedConstraints
    
    @coro
    def setSuccessorCommonProperty(self,prop,val):
        self.succGroupFull.setProperty(prop,val,awaitable=True).get()
    @coro
    def getSuccGroupProxy(self):
        return self.succGroupFull

    # Because charm4py seems to filter **kwargs, pass all arguments to populatePoset in a single dictionary.
    # This avoids having to distinguish between those arguments that are for populatePoset itself and those
    # that are merely passed on to setMethod. This is an implementation distinction not a semantic one: all
    # of these arguments affect the behavior/output of "populatePoset"

    # opts dictionary keys 'clearTable' and 'retrieveFaces' set parameters in populatePoset itself; any
    # other keys are passed as keyword arguments to setMethod
    @coro
    def populatePoset(self, opts={} ):
        if self.populated:
            return
        self.clearTable = 'speed'
        self.retrieveFaces = False
        defaultSettings = ['clearTable','retrieveFaces']
        for ky in defaultSettings:
            if ky in opts:
                setattr(self,ky,opts[ky])
                opts.pop(ky)

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
        thisLevel = [(self.flippedConstraints.root,)]
        posetLen = 1
        levelSizes = [1]
        timedOut = False

        # Send this node into the distributed hash table and check it
        initFut = Future()
        self.distHashTable.initListening(initFut,awaitable=True).get()
        initFut.get()
        self.succGroupFull.startListening(awaitable=True).get()

        boolIdxNoFlip = bytearray(b'\x00') * (self.flippedConstraints.wholeBytes + (1 if self.flippedConstraints.tailBits != 0 else 0))
        for unflipIdx in range(len(thisLevel[0][0])-1,-1,-1):
            boolIdxNoFlip[thisLevel[0][0][unflipIdx]//8] = boolIdxNoFlip[thisLevel[0][0][unflipIdx]//8] | (1<<(thisLevel[0][0][unflipIdx] % 8))
        self.successorProxies[0].hashAndSend([boolIdxNoFlip,thisLevel[0][0]],vertex=(None if self.hashStoreMode != 2 else (self.flippedConstraints.pt,tuple())),ret=True).get()
        
        self.distHashTable.awaitPending(awaitable=True).get()
        # Send a final termination signal:
        self.succGroup.sendAll(-2,awaitable=True).get()
        self.succGroup.closeQueryChannels(awaitable=True).get()
        self.succGroup.flushMessages(ret=True).get()
        
        checkVal = self.distHashTable.levelDone(ret=True).get()
        if not checkVal:
            level = self.N+2
        listenerCount = self.distHashTable.awaitShutdown(ret=True).get()

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
            self.distHashTable.initListening(initFut,awaitable=True).get()
            initFut.get()
            self.succGroupFull.startListening(awaitable=True).get()

            if not self.useGPU:
                self.succGroup.computeSuccessorsNew(ret=True).get()
            else:
                self.succGroup.computeSuccessorsNewGPU(ret=True).get()
            timedOut = any(self.succGroupFull.getTimeout(ret=True).get())
            if timedOut:
                print('Received timeout on level ' + str(level))

            self.distHashTable.awaitPending(awaitable=True).get()
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

            nextLevelSize = self.distHashTable.scheduleNextLevel(clearTable=(self.clearTable == 'memory'),ret=True).get()
            levelSizes.append(nextLevelSize)

            listenerCount = self.distHashTable.awaitShutdown(ret=True).get()

            if self.clearTable == 'speed':
                self.distHashTable.clearHashTable(awaitable=True).get()


            posetLen += nextLevelSize
            # print(posetLen)
            


            # thisLevel = nextLevel
            level += 1

        # Note, this print has to go here because this coroutine is only suspending until checkNodes is set
        statsFut = Future()
        self.succGroupFull.getStats(statsFut)
        stats = statsFut.get()
        stats['levelSizes'] = levelSizes
        print('Total LPs used: ' + str(stats))

        print('Checker returned value: ' + str(checkVal))
        
        # print('Computed a (partial) poset of size: ' + str(len(self.hashTable.keys())))
        print('Computed a (partial) poset of size: ' + str(posetLen))

        if timedOut:
            print('Poset computation timed out...')
        # return [i.iINT for i in self.hashTable.keys()]
        self.populated = True
        return checkVal




class successorWorker(Chare):
    
    def initPEs(self,pes):
        self.posetPElist = pes
        self.timedOut = False

    def initialize(self,N,constraints,timeout):
        self.workInts = []
        self.N = N
        self.flippedConstraints = constraints
        self.constraints = self.flippedConstraints.constraints
        # self.processNodeSuccessors = partial(successorWorker.processNodeSuccessorsFastLP, self, solver='glpk')
        self.processNodeSuccessors = self.thisProxy[self.thisIndex].processNodeSuccessorsFastLP
        self.processNodesArgs = {'solver':'glpk','ret':True}
        # Defaults to glpk, so this empty call is ok:
        self.lp = encapsulateLP.encapsulateLP()
        # self.outChannels = []
        self.clockTimeout = (timeout + time.time()) if timeout is not None else None
        self.timedOut = False
        self.stats = {'LPSolverCount':0, 'xferTime':0, 'numQueries':0, 'successfulQueries':0}
    @coro
    def getTimeout(self):
        return self.timedOut
    
    def setMethod(self,method='fastLP',solver='glpk',findAll=True,useQuery=False,useBounding=False,lpopts={},hashStore='bits',tol=1e-9,rTol=1e-9):
        self.lp.initSolver(solver=solver, opts={'dim':len(self.constraints[0])-1})
        self.useQuery = useQuery
        self.useBounding = useBounding
        self.tol = tol
        self.rTol = rTol
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
        elif method=='fastLP':
            self.processNodeSuccessors = self.thisProxy[self.thisIndex].processNodeSuccessorsFastLP
            self.processNodesArgs = {'solver':solver,'findAll':findAll}
        if len(lpopts) == 0:
            self.processNodesArgs['lpopts'] = lpopts
        self.processNodesArgs['ret'] = True
        self.method = method
        self.solver = solver
        self.findAll = findAll
        self.Hcol0Close = self.tol + self.rTol * np.abs(self.constraints[:,0])
        self.Hcol0CloseVertex = self.constraints[:,0] - self.Hcol0Close
    
    @coro
    def setProperty(self,prop,val):
        setattr(self,prop,val)

    @coro
    def getStats(self, statsFut):
        if charm.myPe() in self.posetPElist:
            self.stats['LPSolverCount'] += self.lp.lpCount
        retVal = defaultdict(int) if not charm.myPe() in self.posetPElist else self.stats
        self.reduce(statsFut,retVal,DictAccum)
    
    @coro
    def getProxies(self):
        return self.thisProxy[self.thisIndex]
    @coro
    def addDestChannel(self, procGroupProxies):
        if not charm.myPe() in self.posetPElist:
            return
        self.numHashWorkers = len(procGroupProxies)
        self.outChannels = [Channel(self, remote=proxy) for proxy in procGroupProxies]
        self.numHashBits = 1
        while self.numHashBits < self.numHashWorkers:
            self.numHashBits = self.numHashBits << 1
        self.hashMask = self.numHashBits - 1
        self.numHashBits -= 1
        # if self.N % 4 == 0:
        #     self.numBytes = self.N/4
        # else:
        #     self.numBytes = int(self.N/4)+1
        # print(self.outChannels)

    @coro
    def addQueryDestChannel(self, procGroupProxies, distHashProxy):
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
        # print(self.outChannels)
    
    @coro
    def closeQueryChannels(self):
        if not charm.myPe() in self.posetPElist:
            return
        for ch in self.queryChannels:
            ch.send(-2)
        if not self.queryMutexChannel is None:
            self.queryMutexChannel.send(-2)

    @coro
    def addFeedbackRateChannelOrigin(self,overlapPElist ):
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
            #print(self.outChannels[k])
            self.outChannels[k].send((self.thisIndex,k))
            #print('Message sent!')
    
    def startListening(self):
        if not charm.myPe() in self.posetPElist:
            return
        for ch in self.outChannels:
            ch.send(-100)

    def hashNode(self,toHash,payload=None,vertex=None):
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
        if payload is not None:
            return ( (hashInt & self.hashMask) % self.numHashWorkers , hashInt >> self.numHashBits, regEncode, charm.myPe(), payload)
        else:
            return ( (hashInt & self.hashMask) % self.numHashWorkers , hashInt >> self.numHashBits, regEncode, charm.myPe(), )
    
    @coro
    def hashAndSend(self,toHash,payload=None,vertex=None):
        val = self.hashNode(toHash,payload=payload,vertex=vertex)
        self.outChannels[val[0]].send(val)
        # print('Trying to hash integer ' + str(nodeInt))
        # retVal = self.thisProxy[self.thisIndex].deferControl(code=5,ret=True).get()
        retVal = self.thisProxy[self.thisIndex].deferControl(ret=True).get()
        # print('Saw defercontrol return the following within HashAndSend ' + str(retVal))
        return retVal
    
    def decodeRegionStore(self,INTrep):
        if type(INTrep) == tuple and len(INTrep) == 2 and type(INTrep[0]) is np.ndarray:
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
            self.rateChannel.send(code)
            control = self.rateChannel.recv() 
            while control > 0:
                control = self.rateChannel.recv()
            if control == -3:
                return False
        return True
    
    @coro
    def query(self, q):
        # print('PE' + str(charm.myPe()) + ' Query to send is ' + str(q))
        self.stats['numQueries'] += 1
        val = self.hashNode(q)
        self.queryChannels[val[0]].send(val)
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
        # print('^^^^^^ Recieved answer to query ' + str(q) + ' of ' + str(retVal))
        if retVal > 0:
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
        fut.send(1)

    #@coro
    def sendAll(self,val):
        if not charm.myPe() in self.posetPElist:
            return
        for ch in self.outChannels:
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
            successorList = [[None,None] for k in range(len(self.workInts))]
            for ii in range(len(successorList)):
                successorList[ii] = self.processNodeSuccessors(self.workInts[ii][0],self.N,self.constraints,**self.processNodesArgs,payload=self.workInts[ii][1:]).get()
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
    def concreteMinHRep(self,H2,constraint_list_in,boolIdxNoFlip,intIdxNoFlip,intIdx,solver='glpk',safe=False):
        
        if len(intIdx) == 0:
            return np.full(0,0,dtype=bool)

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
            if self.useQuery:
                boolIdxNoFlip[intIdx[idx]//8] = boolIdxNoFlip[intIdx[idx]//8] | (1<<(intIdx[idx]%8))
                insertIdx = 0
                while insertIdx < len(intIdxNoFlip) and intIdxNoFlip[insertIdx] < intIdx[idx]:
                    insertIdx += 1
                temp = copy(intIdxNoFlip)
                temp.insert(insertIdx,intIdx[idx])
                # q = self.thisProxy[self.thisIndex].query( bytes(np.packbits(boolIdx,bitorder='little')), ret=True).get()
                q = self.thisProxy[self.thisIndex].query( [boolIdxNoFlip, tuple(temp)], ret=True).get()
                boolIdxNoFlip[intIdx[idx]//8] = boolIdxNoFlip[intIdx[idx]//8] ^ (1<<(intIdx[idx]%8))
                # print('PE' + str(charm.myPe()) + ' Queried table with node ' + str(origInt) + ' and received reply ' + str(q))
                # If the node corresponding to the hyperplane we're about to flip is already in the table
                # then treat it as redundant and skip it (saving the LP)
                if q > 0:
                    continue
            if not safe:
                # Set the extra row to the negation of the pre-relaxed current constraint
                H[-1,:] = -H2[intIdx[idx],:]
            H[offsetIdx,0] += 1
            status, x = self.lp.runLP( \
                    H2[intIdx[idx],1:], \
                    -H[constraint_list,1:], \
                    H[constraint_list,0], \
                    lpopts = {'solver':solver, 'fallback':'glpk'} if solver != 'glpk' else {'solver':'glpk'}, \
                    msgID = str(charm.myPe()) \
                )
            H[offsetIdx,0] -= 1

            if status != 'optimal' and (safe or status != 'primal infeasible') and status != 'dual infeasible':
                print('********************  PE' + str(charm.myPe()) + ' WARNING!!  ********************')
                print('PE' + str(charm.myPe()) + ': Infeasible or numerical ill-conditioning detected at node' )
                print('PE ' + str(charm.myPe()) + ': RESULTS MAY NOT BE ACCURATE!!')
                return [set([]), 0]
            if (safe and -H2[intIdx[idx],1:]@x < H2[intIdx[idx],0]) \
                or (not safe and (status == 'primal infeasible' or np.all(-H2[intIdx[idx],1:]@x - H2[intIdx[idx],0] <= 1e-10))):
                # inequality is redundant, so skip it
                constraint_list[offsetIdx] = False
            else:
                to_keep.append(idx)

        return to_keep

    @coro
    def processNodeSuccessorsFastLP(self,INTrep,N,H,payload=[],solver='glpk',findAll=False,lpopts={}):
        # INTrep = INTrep[0]
        # We assume INTrep is a list of integers representing the hyperplanes that CAN'T be flipped
        # t = time.time()
        INTrep, boolIdxNoFlip, intIdx, intIdxNoFlip = self.decodeRegionStore(INTrep)


        # Flip the un-flippable hyperplanes; this must be undone later
        H[INTrep,:] = -H[INTrep,:]

        
        if findAll:
            intIdx = list(range(self.N))

        
        d = H.shape[1]-1

        
        doBounding = False
        # Don't compute the bounding box if the number of flippable hyperplanes is almost 2*d,
        # since we have to do 2*d LPs just to get the bounding box
        if not findAll and len(intIdx) > 3*d:
            doBounding = True
        # If we want all the faces, we should decide whether to compute the bounding box based on
        # the number N instead:
        if findAll and N > 3*d:
            doBounding = True
        doBounding = doBounding and self.useBounding
        if doBounding:
            #lp = encapsulateLP(solver, opts={'dim':d})
            # Find a bounding box
            bbox = [[] for ii in range(d)]
            ed = np.zeros((d,1))
            for ii in range(d):
                for direc in [1,-1]:
                    ed[ii,0] = direc
                    status, x = self.lp.runLP( \
                        ed.flatten(), \
                        -H[:,1:], H[:,0], \
                        lpopts = {'solver':solver, 'fallback':'glpk'} if solver != 'glpk' else {'solver':'glpk'}, \
                        msgID = str(charm.myPe()) \
                    )
                    ed[ii,0] = 0

                    if status == 'optimal':
                        bbox[ii].append(np.array(x[ii,0]))
                    elif status == 'dual infeasible':
                        bbox[ii].append(-1*direc*np.inf)
                    else:
                        print('********************  PE' + str(charm.myPe()) + ' WARNING!!  ********************')
                        print('PE' + str(charm.myPe()) + ': Infeasible or numerical ill-conditioning detected while computing bounding box!')
                        return [set([]), 0]

            boxCorners = np.array(np.meshgrid(*bbox)).T.reshape(-1,d).T
            constraint_list = np.any(((-H[:,1:] @ boxCorners) - H[:,0].reshape((-1,1))) >= -1e-07,axis=1)
        else:
            constraint_list = None

        
        faces = self.thisProxy[self.thisIndex].concreteMinHRep(H,constraint_list,boolIdxNoFlip,intIdxNoFlip,intIdx,solver=solver,safe=False,ret=True).get()

        successors = []
        for i in faces:
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
                        [ copy(boolIdxNoFlip), tuple(temp) ]
                    )
                boolIdxNoFlip[intIdx[i]//8] = boolIdxNoFlip[intIdx[i]//8] ^ 1<<(intIdx[i] % 8)
                # self.conversionTime += time.time() - t
                t = time.time()
                cont = self.thisProxy[self.thisIndex].hashAndSend(successors[-1],ret=True).get()
                self.stats['xferTime'] += time.time() - t

                if not cont:
                    return [successors, -1]
        
        # facesInt = np.full(self.N,0,dtype=bool)
        sel = tuple(np.array(intIdx,dtype=np.uint64)[faces].tolist())
        # facesInt[sel] = np.full(len(sel),1,dtype=bool)

        # Undo the flip we did before, since it affects a referenced (as opposed to copied) array:
        H[INTrep,:] = -H[INTrep,:]

        # return [successors, bytes(np.packbits(facesInt,bitorder='little'))]            
        return [[], sel]            

      



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