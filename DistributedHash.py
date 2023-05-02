import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
from numpy import may_share_memory
import time
import itertools
import random
from copy import copy
import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict
from numba.np.unsafe.ndarray import to_fixed_tuple
from functools import partial
from region_helpers import hashNodeBytes

XFER_CHUNK_SIZE = 1000

class Node():

    def __init__(self,localProxy, storePe, parentChare, nodeEqualityFn, lsb,msb,nodeBytes, originPe, face, witness, *args):
        self.lsbHash = lsb
        self.msbHash = msb
        self.nodeBytes = nodeBytes
        self.localProxy = localProxy
        self.storePe = storePe
        self.parentChare = parentChare
        self.originPe = originPe
        self.face = set(face)
        self.witness = witness
        self.nodeEqualityFn = nodeEqualityFn
        self.payload = args[0] if len(args) > 0 else tuple()

    def __hash__(self):
        return self.msbHash

    def __eq__(self,other):
        if type(other) == type(self.nodeBytes):
            return self.nodeEqualityFn(self.nodeBytes, other)
        elif isinstance(other,Node):
            return self.nodeEqualityFn(self.nodeBytes, other.nodeBytes)

# Vertex Node equality check
@njit( \
    types.int64[::1] \
    ( \
        types.float64[:,::1], \
        types.float64[::1], \
        types.float64[:,::1], \
        types.int64[::1] \
    ), \
    cache=True \
)
def vertexNodeDecode(H,H0close,aSol,aList):
    diff = (-H[:,1:] @ aSol).flatten() - H[:,0]
    flipIdxsA = ( diff > -H0close).flatten().astype(np.bool8)
    activeFlips = np.nonzero((np.abs(diff) <= H0close).flatten())[0]
    flipIdxsA[activeFlips] = np.zeros(activeFlips.shape,dtype=np.bool8)
    flipIdxsA[aList] = np.ones(aList.shape,dtype=np.bool8)
    return np.nonzero(flipIdxsA)[0]

def vertexNodeEquality(H,H0close,wholeBytes,tailBits,a,b):
    result = None
    for reg in [a,b]:
        if type(reg) == tuple and len(reg) == 2 and type(reg[1]) == tuple:
            INTrep = tuple(vertexNodeDecode(H,H0close,reg[0],np.array(reg[1],dtype=np.int64)).tolist())
        elif type(reg) == bytearray:
            INTrep = []
            for bIdx in range(wholeBytes + (1 if tailBits != 0 else 0)):
                for bitIdx in range(8 if bIdx < wholeBytes else tailBits):
                    if boolIdxNoFlip[bIdx] & ( 1 << bitIdx):
                        INTrep.append(8*bIdx + bitIdx)
            INTrep = tuple(INTrep)
        else:
            INTrep = reg

        result = INTrep if result is None else (result == INTrep)

    return result

def Join(contribs):
    return list(itertools.chain.from_iterable(contribs))
Reducer.addReducer(Join)

class HashWorker(Chare):

    def __init__(self,nodeConstructor,localVarGroup,parentProxy,pes,feederPEs,overlapPEs):
        self.hashPElist = pes
        self.feederPElist = feederPEs
        self.overlapPElist = overlapPEs
        self.hashChannelsHashEnd = []
        self.level = -1
        self.levelList = []
        self.levelDone = True
        self.table = {}
        self.nodeConstructor = nodeConstructor
        self.nodeCalls = 0
        self.maxNodeYields = 1
        callIdx = 0
        for checkCall in ['init','update','check']:
            call = getattr(self.nodeConstructor,checkCall,None)
            if callable(call):
                self.nodeCalls += 1 << callIdx
            callIdx += 1
        self.localVarGroup = localVarGroup
        self.parentProxy = parentProxy
        # self.parentChannel = Channel(self,remote=self.parentProxy)
        self.loopback = Channel(self,remote=self.thisProxy[self.thisIndex])
        self.queryLoopback = Channel(self,remote=self.thisProxy[self.thisIndex])
        self.controlLoopback = Channel(self,remote=self.thisProxy[self.thisIndex])
        self.nodeEqualityFn = lambda x,y: (x == y)
        self.hashStoreMode = 1
        self.enumChannelsHashEnd = {}
        self.deferLock = False
        self.msgCounter = 0
        self.processedNodeCounter = 0
        self.hashedNodeCount = 0
        #print(self.thisIndex)
    @coro
    def _NOOP_(self):
        pass
    @coro
    def setConstraint(self,hashStoreMode=1):
        self.hashStoreMode = 1

    @coro
    def updateNodeEqualityFn(self,fn=None,nodeType='standard',tol=1e-9,rTol=1e-9,H=None):
        if fn is not None:
            self.nodeEqualityFn = fn
        else:
            if nodeType == 'standard':
                self.nodeEqualityFn = lambda x,y: (x == y)
            elif nodeType == 'vertex':
                self.H0close = tol + rTol * np.abs(H[:,0])
                self.H = H
                self.wholeBytes = (len(self.H0close) + 7) // 8
                self.tailBits = len(self.H0close) - 8*(len(self.H0close) // 8)
                self.nodeEqualityFn = partial(vertexNodeEquality,self.H,self.H0close,self.wholeBytes,self.tailBits)
                # self.nodeEqualityFn = lambda x,y: ( np.all(np.isclose(x[0],y[0],rtol=rTol,atol=tol)) and (x[1] == y[1]) )
    @coro
    def getProxies(self):
        return self.thisProxy[self.thisIndex]
    def setPos(self,pos):
        self.pos = pos

    @coro
    def initHashChannelHashEnd(self,feederProxies):
        if not charm.myPe() in self.hashPElist:
            return
        self.feederProxies = feederProxies
        self.numFeederWorkers = len(feederProxies)
        self.hashChannelsHashEnd = [Channel(self, remote=proxy) for proxy in feederProxies]
        self.status = {}
        self.messages = {}
        self.workerDone = {}
        self.localListenerOnline = {}
        for ch in self.hashChannelsHashEnd:
            self.status[ch] = 0
            self.messages[ch] = {'msg':None, 'fut':None}
            self.workerDone[ch] = None
            self.localListenerOnline[ch] = False


    @coro
    def initQueryChannelHashEnd(self,feederProxies):
        if not charm.myPe() in self.hashPElist:
            return
        # Call this **after** establishing the hashing channels
        # self.feederProxies = feederProxies
        # self.numFeederWorkers = len(feederProxies)
        self.queryChannelsHashEnd = [Channel(self, remote=proxy) for proxy in feederProxies]
        self.queryStatus = {}
        self.queryMessages = {}
        self.queryDone = {}
        self.queryListenerOnline = {}
        for ch in self.queryChannelsHashEnd:
            self.queryStatus[ch] = 0
            self.queryMessages[ch] = {'msg':None, 'fut':None}
            self.queryDone[ch] = None
            self.queryListenerOnline[ch] = False

    # The following two methods allow a DistributedHash to function as a "feeder" to **another** DistributedHash!
    # To do: implement the feedback channel from the other DistributedHash...
    @coro
    def initHashChannel(self, procGroupProxies):
        if not charm.myPe() in self.hashPElist:
            return
        self.numHashWorkers = len(procGroupProxies)
        self.hashChannels = [Channel(self, remote=proxy) for proxy in procGroupProxies]
        self.numHashBits = 1
        while self.numHashBits < self.numHashWorkers:
            self.numHashBits = self.numHashBits << 1
        self.hashMask = self.numHashBits - 1
        self.numHashBits -= 1
        # print(f'Finished Executing initHashChannels on first distributed hash')
        # if self.N % 4 == 0:
        #     self.numBytes = self.N/4
        # else:
        #     self.numBytes = int(self.N/4)+1
        # print(self.hashChannels)
    @coro
    def initQueryChannel(self, procGroupProxies, distHashProxy):
        if not charm.myPe() in self.hashPElist:
            return
        # self.numHashWorkers = len(procGroupProxies)
        self.queryChannels = [Channel(self, remote=proxy) for proxy in procGroupProxies]
        self.queryMutexChannel = None
        if not self.rateChannel is None:
            self.queryMutexChannel = Channel(self, remote=distHashProxy)
    @coro
    def closeQueryChannels(self):
        if not charm.myPe() in self.hashPElist:
            return
        for ch in self.queryChannels:
            ch.send(-2)
        if not self.queryMutexChannel is None:
            self.queryMutexChannel.send(-2)

    @coro
    def initRateChannel(self,overlapPElist ):
        self.rateChannel = None
        self.overlapPElistAsFeeder = overlapPElist
        if not charm.myPe() in self.hashPElist:
            return
        if charm.myPe() in overlapPElist:
            self.rateChannel = Channel(self,remote=overlapPElist[charm.myPe()][1])
        # self.feedbackChannel = Channel(self,remote=proxy)

    @coro
    def startListening(self):
        if not charm.myPe() in self.hashPElist:
            return
        for ch in self.hashChannels:
            ch.send(-100)

    @coro
    def deferControl(self, code=1):
        if not self.rateChannel is None:
            while self.deferLock:
                self.thisProxy[self.thisIndex]._NOOP_(awaitable=True).get()
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
    def hashNode(self,toHash,payload=None):
        # hashInt = int(posetFastCharm_numba.hashNodeBytes(np.array(toHash[0],dtype=np.uint8)))
        # hashInt = hashNodeBytes(np.array(toHash[0],dtype=np.uint8))
        hashInt = hashNodeBytes(toHash[0])
        if self.hashStoreMode == 0:
            regEncode = toHash[0]
        elif self.hashStoreMode == 1:
            regEncode = tuple(toHash[1])
        else:
            # default to tuple mode
            regEncode = tuple(toHash[1])
        if len(toHash) >= 3:
            face = toHash[2]
        else:
            face = tuple()
        if len(toHash) >= 4:
            witness = toHash[3]
        else:
            witness = None
        if payload is not None:
            return ( (hashInt & self.hashMask) % self.numHashWorkers , hashInt >> self.numHashBits, regEncode, charm.myPe(), face, witness, payload)
        else:
            return ( (hashInt & self.hashMask) % self.numHashWorkers , hashInt >> self.numHashBits, regEncode, charm.myPe(), face, witness )

    @coro
    def hashAndSend(self,toHash,payload=None):
        self.hashedNodeCount += 1
        val = self.hashNode(toHash,payload=payload)
        self.hashChannels[val[0]].send(val)
        # print('Trying to hash integer ' + str(val))
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
    def sendAll(self,val):
        if not charm.myPe() in self.hashPElist:
            return
        for ch in self.hashChannels:
            ch.send(val)

    @coro
    def flushMessages(self):
        if not charm.myPe() in self.overlapPElistAsFeeder:
            return
        self.rateChannel.send(2)
    @coro
    def closeQueryChannels(self):
        if not charm.myPe() in self.hashPElist:
            return
        for ch in self.queryChannels:
            ch.send(-2)
        if not self.queryMutexChannel is None:
            self.queryMutexChannel.send(-2)
    @coro
    def addFeedbackChannel(self,proxy):
        if not charm.myPe() in self.hashPElist:
            return
        self.feedbackChannel = Channel(self,remote=proxy)
    @coro
    def initRateChannelHashEnd(self,overlapPElist):
        self.rateChannelHashEnd = None
        self.termProxy = None
        self.overlapPElist = overlapPElist
        if not charm.myPe() in self.hashPElist:
            return
        if charm.myPe() in overlapPElist:
            self.rateChannelHashEnd = Channel(self,remote=overlapPElist[charm.myPe()][0])
        elif len(overlapPElist) > 0:
            self.termProxy = self.overlapPElist[ list(self.overlapPElist.keys())[0] ][0]


    @coro
    def localListener(self,ch,chIdx):
        self.initiatedNodeProc = False
        validInput = False
        while True:
            # print('Listener started')
            # charm.wait([ch])
            # sig = str(random.random())
            # before = [self.messages[chIt]['msg'] for chIt in self.hashChannelsHashEnd]
            val = ch.recv()
            # print(f'PE{charm.myPe()} MSG:{self.msgCounter}: Recieved val ' + str(val) + 'on PE ' + str(charm.myPe()))
            # self.msgCounter += 1
            # Make sure we don't start reading until the previous level/poset was done
            if not validInput:
                if val == -100:
                    validInput = True
                continue
            # print(val)
            self.messages[ch]['msg'] = val
            ackFut = Future()
            self.messages[ch]['fut'] = ackFut
            # print(f'PE{charm.myPe()} MSG:{self.msgCounter-1}: {self.messages}')
            self.loopback.send(chIdx)
            # if not self.initiatedNodeProc:
            #     print('----'*(charm.myPe()+1) + '>>  PE'+str(charm.myPe())+'LocalListener ' + sig + ' -- Signaling to listen()' )
            if not self.initiatedNodeProc:
                self.initiatedNodeProc = True
                self.controlLoopback.send(1)
            # self.parentProxy.sendFeedbackMessage(-1*charm.myPe())
            # print('----'*(charm.myPe()+1) + '>>  PE'+str(charm.myPe())+'LocalListener ' + sig + ' -- initiatedNodeProc = ' + str(self.initiatedNodeProc) + ' Before: ' + str(before) + ' After: ' + str([self.messages[chIt]['msg'] for chIt in self.hashChannelsHashEnd]))
            ackFut.get()
            # self.parentProxy.sendFeedbackMessage(charm.myPe())
            self.messages[ch]['msg'] = None
            self.messages[ch]['fut'] = None
            if val == -3 or val == -2:
                # print('----'*(charm.myPe()+1) + '>>  PE'+str(charm.myPe())+'LocalListener ' + sig + ' -- TERMINATING!')
                break
        return 1

    @coro
    def localQueryListener(self,ch,chIdx):
        self.initiatedQueryProc = False
        while True:
            # print('Listener started')
            # charm.wait([ch])
            # sig = str(random.random())
            # before = [self.queryMessages[chIt]['msg'] for chIt in self.queryChannelsHashEnd]
            val = ch.recv()
            # print(val)
            self.queryMessages[ch]['msg'] = val
            ackFut = Future()
            self.queryMessages[ch]['fut'] = ackFut
            self.queryLoopback.send(chIdx)
            # if not self.initiatedNodeProc:
                # print('===='*(charm.myPe()+1) + '>>  PE'+str(charm.myPe())+'QUERYLocalListener ' + sig + ' -- Signaling to listen()' )
            if not self.initiatedQueryProc:
                self.initiatedQueryProc = True
                self.controlLoopback.send(2)
            # self.parentProxy.sendFeedbackMessage(-1*charm.myPe())
            # print('===='*(charm.myPe()+1) + '>>  PE'+str(charm.myPe())+'QUERYLocalListener ' + sig + ' -- initiatedQueryProc = ' + str(self.initiatedNodeProc) + ' Before: ' + str(before) + ' After: ' + str([self.queryMessages[chIt]['msg'] for chIt in self.queryChannelsHashEnd]))
            ackFut.get()
            # self.parentProxy.sendFeedbackMessage(charm.myPe())
            self.queryMessages[ch]['msg'] = None
            self.queryMessages[ch]['fut'] = None
            if val == -2:
                # print('===='*(charm.myPe()+1) + '>>  PE'+str(charm.myPe())+'QUERYLocalListener ' + sig + ' -- TERMINATING!')
                break
        return 1

    @coro
    def initListen(self,fut,queryReturnInfo=False):
        self.processedNodeCounter = 0
        self.queryReturnInfo = queryReturnInfo
        if not charm.myPe() in self.hashPElist:
            return
        # Make sure all of the loopback channels are empty:
        for loopbackIt in [self.loopback, self.queryLoopback, self.controlLoopback]:
            loopbackIt.send(-100)
            while True:
                m = loopbackIt.recv()
                if m == -100:
                    break
        for ch in self.hashChannelsHashEnd:
            self.status[ch] = 0
            self.messages[ch] = {'msg':None, 'fut':None}
            self.workerDone[ch] = None
        self.listenerStatus = []
        for k in range(len(self.hashChannelsHashEnd)):
            self.listenerStatus.append( self.thisProxy[self.thisIndex].localListener(self.hashChannelsHashEnd[k],k, ret=True) )
            self.workerDone[self.hashChannelsHashEnd[k]] = Future()

        for ch in self.queryChannelsHashEnd:
            self.queryStatus[ch] = 0
            self.queryMessages[ch] = {'msg':None, 'fut':None}
            self.queryDone[ch] = None
        for k in range(len(self.queryChannelsHashEnd)):
            self.listenerStatus.append( self.thisProxy[self.thisIndex].localQueryListener(self.queryChannelsHashEnd[k],k,ret=True) )
            self.queryDone[self.queryChannelsHashEnd[k]] = Future()

        # print('Done initListen on PE ' + str(charm.myPe()))
        self.level += 1
        self.levelList = []
        self.termCount = 0
        self.levelDone = False
        fut.send(1)



    @coro
    def listen(self):
        # print('Started main listener')
        free = False
        pendingChecks = False
        pendingQueries = False
        processOnly = False
        selfQuery = False
        queryOnly = False
        yieldCount = 0
        cnt = 1
        msgCount = {}
        for ch in self.hashChannelsHashEnd:
            msgCount[ch] = 0
        while any([self.queryStatus[ch] > -2 for ch in self.queryChannelsHashEnd]) or \
            any([self.status[ch] > -2 for ch in self.hashChannelsHashEnd]): # or not free:
            # traceSig = random.random()
            # peSig = 2*charm.myPe()*'    ' + 'PE_' + str(charm.myPe()) + 2*(charm.numPes()+4 - charm.myPe())*' ' + '  '
            # indent = cnt*8*' '
            # prefix = peSig + indent + str(traceSig)
            # print(prefix + ' Hash worker on PE ' + str(charm.myPe()) + ' START STATE ---> [pendingChecks, pendingQueries] = ' + str([pendingChecks, pendingQueries]) +  \
            #     'Node Listener status: ' + str([self.status[chIt] for chIt in self.hashChannelsHashEnd]) + ' Query Listener status: ' + str([self.queryStatus[chIt] for chIt in self.queryChannelsHashEnd]) + ' ' \
            #         + 'Node messages buffered: ' + str([self.messages[ch]['msg'] for ch in self.hashChannelsHashEnd]) + ' Query messages buffered: ' + str([self.queryMessages[ch]['msg'] for ch in self.queryChannelsHashEnd])  + ' ' \
            #             + 'Message buffers: ' + str([self.messages[ch]['fut'] for ch in self.hashChannelsHashEnd]) + str([self.queryMessages[ch]['fut'] for ch in self.queryChannelsHashEnd]))
            # If we're running on the same PE as a feeder worker, and he hasn't signaled "free run mode" (i.e. he has no more work to do)
            # then wait for the feeder to transfer control
            if not self.rateChannelHashEnd is None and not free:
                control = self.rateChannelHashEnd.recv()
                # print(prefix + ' RECEIVED RELEASE CONTROL SIGNAL ' + str(control) + ' on PE ' + str(charm.myPe()) + ' ')
                if control == 2:
                    free = True
                    queryOnly = False
                    selfQuery = False
                    processOnly = True
                if control <= 0:
                    self.rateChannelHashEnd.send(control)
                    continue
                if control == 5:
                    queryOnly = False
                    selfQuery = False
                    processOnly = True
                if control == 1:
                    queryOnly = False
                    selfQuery = False
                    processOnly = False
                if control == 4:
                    queryOnly = True
                    selfQuery = False
                    processOnly = False
                if control == 3:
                    queryOnly = True
                    selfQuery = True
                    processOnly = False
                if all([self.status[ch] <= -2 for ch in self.hashChannelsHashEnd]) and \
                        all([self.queryStatus[ch] <= -2 for ch in self.queryChannelsHashEnd]):
                    self.rateChannelHashEnd.send(min([self.status[ch] for ch in self.hashChannelsHashEnd]))
                    break
                # print('Received control of ' + str(control) + ' on PE ' + str(charm.myPe()))
                if all([self.messages[ch]['fut'] is None for ch in self.hashChannelsHashEnd]) and \
                    all([self.queryMessages[ch]['fut'] is None for ch in self.queryChannelsHashEnd]):
                    self.rateChannelHashEnd.send(-1)
                    continue
                if queryOnly and not selfQuery and all([self.queryMessages[ch]['fut'] is None for ch in self.queryChannelsHashEnd]):
                    self.rateChannelHashEnd.send(-1)
                    continue

            # print(prefix + ' Starting trace on PE ' + str(charm.myPe()) + ' -------------------------')
            pendingWork = [3,3]
            if not processOnly:
                pendingWork[0] = self.controlLoopback.recv()
                if ((not pendingChecks and pendingWork[0] > 1) and any([not self.messages[ch]['fut'] is None for ch in self.hashChannelsHashEnd])) or \
                        pendingWork[0] == 1 and any([not self.queryMessages[ch]['fut'] is None for ch in self.queryChannelsHashEnd]):
                    pendingWork[1] = self.controlLoopback.recv()
                if 1 in pendingWork:
                    pendingChecks = True
                    self.initiatedNodeProc = False
                if 2 in pendingWork:
                    pendingQueries = True
                    self.initiatedQueryProc = False

            # print(prefix + ' Received control signal ' + str(pendingWork[0]) + ' on PE ' + str(charm.myPe()))
            # print(prefix + ' [pendingChecks, pendingQueries] = ' + str([pendingChecks, pendingQueries]) + ' on PE ' + str(charm.myPe()))
            # print(prefix + ' Hash worker on PE ' + str(charm.myPe()) + ' CONTROL STATE ---> pendingWork=' + str(pendingWork) + ' ' +  \
            #     'Node Listener status: ' + str([self.status[chIt] for chIt in self.hashChannelsHashEnd]) + ' Query Listener status: ' + str([self.queryStatus[chIt] for chIt in self.queryChannelsHashEnd]) + ' ' \
            #         + 'Node messages buffered: ' + str([self.messages[ch]['msg'] for ch in self.hashChannelsHashEnd]) + ' Query messages buffered: ' + str([self.queryMessages[ch]['msg'] for ch in self.queryChannelsHashEnd])  + ' ' \
            #             + 'Message buffers: ' + str([self.messages[ch]['fut'] for ch in self.hashChannelsHashEnd]) + str([self.queryMessages[ch]['fut'] for ch in self.queryChannelsHashEnd]))


            # Count the number of times we execute when there are queries pending:
            if pendingChecks or not pendingQueries or all([self.queryStatus[chIt] == -2 for chIt in self.queryChannelsHashEnd]):
                yieldCount += 1

            # print(prefix + ' Hash worker on PE ' + str(charm.myPe()) + ' pendingWork=' + str(pendingWork) + ' ' + \
            #    str([self.messages[ch]['msg'] for ch in self.hashChannelsHashEnd]) + ' ' + \
            #        str([self.queryMessages[ch]['msg'] for ch in self.queryChannelsHashEnd])  + ' ' + str(self.status))
            # Respond to hash table queries
            chList = []
            answeredSelf = False
            if pendingQueries:
                numPending = sum([not self.queryMessages[ch]['fut'] is None for ch in self.queryChannelsHashEnd])
                # print('numPending on PE ' + str(charm.myPe()) + ' is ' + str(numPending))
                i = 0
                while i < numPending:
                    chIdx = self.queryLoopback.recv()
                    ch = self.queryChannelsHashEnd[chIdx]
                    msg = self.queryMessages[ch]
                    chList.append(ch)
                    val = msg['msg']
                    # print(prefix + ' Processing QUERY message ' + str(val) + ' on PE ' + str(charm.myPe()))
                    if type(val) is tuple and len(val) >= 3:
                        if (not charm.myPe() in self.overlapPElist) or (not selfQuery or charm.myPe() == chIdx):
                            answeredSelf = True
                        newNode = self.nodeConstructor(self.localVarGroup, charm.myPe(), self, self.nodeEqualityFn, *val)
                        if newNode in self.table:
                            # print('Responding to query ' + str(val) + ' on channel ' + str(chIdx))
                            nd = self.table[newNode]['ptr']
                            self.queryChannelsHashEnd[chIdx].send((1,) if not self.queryReturnInfo else (1, nd.face, nd.witness, nd.payload))
                        else:
                            # print('Responding to query ' + str(val) + ' on channel ' + str(chIdx))
                            self.queryChannelsHashEnd[chIdx].send((-1,))
                    elif val < 0:
                        answeredSelf = True
                        # for chIt in self.queryChannelsHashEnd:
                        self.queryStatus[ch] = -2
                        self.queryDone[ch].send(True)
                    else:
                        print('PE' + str(charm.myPe()) + ' : Unexpected query received. Query was ' + str(val))
                    i += 1
                    # Ensure we catch any messages arriving at the local listener while we were receiving on the queryLoopback channel
                    numPending = sum([not self.queryMessages[ch]['fut'] is None for ch in self.queryChannelsHashEnd])
                    # If this is a self query and it hasn't been answered yet, repeat the loop until it is
                    if selfQuery and not answeredSelf and i == numPending:
                        numPending += 1

                for ch in chList:
                    # We're all done with this message, so report back
                    localFut = self.queryMessages[ch]['fut']
                    self.queryMessages[ch]['fut'] = None
                    if not localFut is None:
                        localFut.send(1)


                pendingQueries = False

            # Now take care of any hashing/checking any pending nodes supplied by (any) feeder worker
            chList = []
            # Don't actually process hash requests until we have yielded at least this many times
            # to the query processing -- this effectively prioritizes processing queries, and should
            # reduce the processing latency for them at the expense of latency for new node hashing/checking
            # print(prefix + ' PendingChecks is ' + str(pendingChecks) + ' and pendingQueries is ' + str(pendingQueries))
            if pendingChecks and (not queryOnly or free) and yieldCount >= self.maxNodeYields:
                # print(prefix + ' Pending messages on PE ' + str(charm.myPe()) + ' are ' + str(self.messages))
                numPending = sum([not self.messages[ch]['fut'] is None for ch in self.hashChannelsHashEnd])
                # print(prefix + 'numPending nodes is ' + str(numPending) + 'on PE ' + str(charm.myPe()))
                i = 0
                while i < numPending:
                    chIdx = self.loopback.recv()
                    # print(prefix + ' Recieved a chIdx of ' + str(chIdx) + ' on PE ' + str(charm.myPe()))
                    ch = self.hashChannelsHashEnd[chIdx]
                    msg = self.messages[ch]
                    # print(prefix + ' Processing node ' + str(msg) + ' on PE ' + str(charm.myPe()))
                    chList.append(ch)
                    val = msg['msg']
                    msgCount[ch] += 1
                    if val == -3:
                        # for ch in self.hashChannelsHashEnd:
                        if self.status[ch] != -2 and self.status[ch] != -3:
                            self.workerDone[ch].send(True)
                        self.status[ch] = -3
                        self.localVarGroup.setSkip(True)
                    elif val == -4:
                        # msgFalseSet = False
                        # for ch in self.hashChannelsHashEnd:
                        if self.status[ch] != -2 and self.status[ch] != -3:
                            self.workerDone[ch].send(False)
                            msgFalseSet = True
                        self.status[ch] = -3
                        # if not msgFalseSet:
                        #     pass
                        #     # print('**WARNING**: \'False\' early termination signal was received, but didn\'t set any return values. Possible synchronization issue!')
                    elif val == -2:
                        if self.status[ch] != -2 and self.status[ch] != -3:
                            self.workerDone[ch].send(True)
                            self.status[ch] = -2
                        # self.workerDone[ch].send(True)
                        if all([self.status[chIt] <= -2 for chIt in self.hashChannelsHashEnd]):
                            self.levelDone = True
                    elif type(val) == tuple and len(val) >= 3:
                        newNode = self.nodeConstructor(self.localVarGroup, charm.myPe(), self, self.nodeEqualityFn, *val)
                        if self.nodeCalls & 1:
                            newNode.init()
                        if not newNode in self.table:
                            self.table[newNode] = {'checked':False, 'ptr':newNode}
                            # self.levelList.append((val[2],*newNode.payload))
                            self.levelList.append(newNode)
                            # Check node here:
                            if self.nodeCalls & 4 and not newNode.check(): # If result of node check is False return False on all the workerDone Futures
                                    if self.status[ch] != -2 and self.status[ch] != -3 and not self.workerDone[ch] is None:
                                        self.workerDone[ch].send(False)
                                    self.status[ch] = -3
                                    # self.parentProxy.sendFeedbackMessage(charm.numPes()+1)
                                    self.levelDone = True
                        elif self.nodeCalls & 2:
                            self.table[newNode]['ptr'].update(*val)
                    # If self.status[ch] == -2 or -3, we know we're supposed to shutdown so ignore any other messages
                    elif self.status[ch] != -2 and self.status[ch] != -3 and not msg['fut'] is None:
                        print(self.status)
                        print(msg)
                        print(val)
                        print('Received unexpected message ' + str(val) + ' on hash worker ' + str(self.thisIndex))

                    # if len(chList) == numPending:
                    #     # Done processing the number of buffered nodes we saw at first, so break out of the while
                    #     # loop to reset those buffers
                    #     break
                    # print('Hash table on PE ' + str(charm.myPe()) + str(self.table))
                    i += 1
                    numPending = sum([not self.messages[ch]['fut'] is None for ch in self.hashChannelsHashEnd])
                    yieldCount = 0
                    pendingChecks = False

                for ch in chList:
                    # We're all done with this message, so report back
                    localFut = self.messages[ch]['fut']
                    self.messages[ch]['fut'] = None
                    if not localFut is None:
                        localFut.send(1)



            # Release the feeder to get back to work:
            processOnly = False
            retControl = -3 if any([self.status[ch] == -3 for ch in self.hashChannelsHashEnd]) else -1
            if not self.rateChannelHashEnd is None and not free:
                self.rateChannelHashEnd.send(retControl)
            elif not self.termProxy is None and retControl == -3:
                self.termProxy.sendAll(-3, awaitable=True).get()
            # print(prefix + ' Hash worker on PE ' + str(charm.myPe()) + ' FINAL STATE ---> pendingWork=' + str(pendingWork) + ' ' +  \
            #     'Node Listener status: ' + str([self.status[chIt] for chIt in self.hashChannelsHashEnd]) + ' Query Listener status: ' + str([self.queryStatus[chIt] for chIt in self.queryChannelsHashEnd]) + ' ' \
            #         + 'Node messages buffered: ' + str([self.messages[ch]['msg'] for ch in self.hashChannelsHashEnd]) + ' Query messages buffered: ' + str([self.queryMessages[ch]['msg'] for ch in self.queryChannelsHashEnd])  + ' ' \
            #             + 'Message buffers: ' + str([self.messages[ch]['fut'] for ch in self.hashChannelsHashEnd]) + str([self.queryMessages[ch]['fut'] for ch in self.queryChannelsHashEnd]))
            # print(prefix + ' Finished trace on PE ' + str(charm.myPe()) + ' ----------------------')
            cnt += 1
        # print('Shutting down main listener on PE ' + str(charm.myPe()))
        return 1

    @coro
    def awaitLevel(self):
        retVal = all([self.workerDone[ch].get() for ch in self.hashChannelsHashEnd])
        self.levelDone = True
        return retVal
    @coro
    def awaitQueries(self):
        return all([self.queryDone[ch].get() for ch in self.queryChannelsHashEnd])
    @coro
    def awaitListenerShutdown(self, shutdownFut):
        cnt = 0
        if charm.myPe() in self.hashPElist:
            for fut in charm.iwait(self.listenerStatus):
                cnt += fut.get()
            self.listenerStatus = []
        self.reduce(shutdownFut, cnt, Reducer.sum)

    @coro
    def getLevelList(self, levelListFut):
        if self.levelDone:
            self.reduce(levelListFut, [(nd.nodeBytes, *nd.payload) for nd in self.levelList], Reducer.Join)
            # self.levelList = []
            # self.table = {}
        else:
            print('Warning: tried to retrieve level list before level was done!')
            self.reduce(levelListFut, [], Reducer.Join)
    @coro
    def getLevelSizes(self):
        return len(self.levelList)
    @coro
    def getTableLen(self):
        return len(self.table)

    @coro
    def clearHashTable(self):
        self.levelList = []
        self.table = {}
    @coro
    def getTable(self):
        return [(ky.nodeBytes, ky.face, ky.witness, ky.payload) for ky in self.table.keys()]
    @coro
    def resetLevelCount(self):
        self.level=-1
    # @coro
    # def receiveOnNodeChannel(self):
    #     if not charm.myPe() in self.hashPElist:
    #         return
    #     count = 0
    #     for i in charm.iwait([self.hashChannelsHashEnd[ch].recv() for ch in self.hashChannelsHashEnd]):
    #         count += 1
    #     return count
    @coro
    def distributeTableChunk(self,feederPEoffset,retFut,clearTable=True):
        while len(self.levelList) > 0:
            chunkSize = min(XFER_CHUNK_SIZE * len(self.feederPElist), len(self.levelList))
            xferDone = [Future() for _ in range(len(self.feederPElist))]
            for feederPEidx in range(len(self.feederPElist)):
                idx = (feederPEidx + feederPEoffset) % len(self.feederPElist)
                self.feederProxies[idx].appendToWorkList([(nd.nodeBytes, nd.payload) for nd in self.levelList[feederPEidx:chunkSize:len(self.feederPElist)]],xferDone[feederPEidx])
            cnt = 0
            for fut in charm.iwait(xferDone):
                cnt += fut.get()
            if clearTable:
                for idx in range(chunkSize):
                    self.table.pop(self.levelList[idx])
            self.levelList = self.levelList[chunkSize:]
        retFut.send(1)

    @coro
    def registerEnumHashEnd(self, remChare, chareKey):
        if chareKey in self.enumChannelsHashEnd:
            print(f'Error: coordinating chare is already registered.')
            return
        self.enumChannelsHashEnd[chareKey] = {'data':Channel(self,remote=remChare),'ctrl':Channel(self,remote=remChare),'lock':False}

    @coro
    def enumListener(self, chareKey, statusFut):
        if not chareKey in self.enumChannelsHashEnd:
            print(f'Error: {chareKey} is not registered...')
            statusFut.send(-1)
            return False
        if self.enumChannelsHashEnd[chareKey]['lock']:
            print(f'Error: {chareKey} already has a lock!')
            statusFut.send(-1)
        self.enumChannelsHashEnd[chareKey]['lock'] = True
        statusFut.send(self.hashPElist.index(charm.myPe()))
        ctrlChan = self.enumChannelsHashEnd[chareKey]['ctrl']
        dataChan = self.enumChannelsHashEnd[chareKey]['data']
        term = False
        for nd in self.table.keys():
            ctrlVal = ctrlChan.recv()
            if ctrlVal > 0:
                ptr = self.table[nd]['ptr']
                dataChan.send((ptr.lsbHash, ptr.msbHash, ptr.nodeBytes, ptr.originPe, ptr.face, ptr.witness, ptr.payload))
            else:
                term = True
                dataChan.send(None)
                break
        if not term:
            ctrlVal = ctrlChan.recv()
            dataChan.send(None)
        # print(f'Shutting down enumListener on PE {charm.myPe()}... Reason: ' + ('table done' if not term else 'shutdown requested'))
        self.enumChannelsHashEnd[chareKey]['lock'] = False



class DistHash(Chare):
    @coro
    def __init__(self, feederGroup, nodeConstructor, localVarGroup, hashPEs, posetPEs, feederSpec):
        self.feederGroup = feederGroup
        self.posetPEs = posetPEs
        self.posetPElist = list(itertools.chain.from_iterable( \
               [list(range(r[0],r[1],r[2])) for r in self.posetPEs] \
            ))
        self.nodeConstructor = nodeConstructor
        if self.nodeConstructor is None:
            self.nodeConstructor = Node
        self.localVarGroup = localVarGroup
        self.hashPEs = hashPEs
        self.hashPElist = list(itertools.chain.from_iterable( \
               [list(range(r[0],r[1],r[2])) for r in self.hashPEs] \
            ))

        # Get a list of proxies for all memembers of the feeder group:
        secs = self.feederGroup.getProxies(ret=True).get()
        self.feederProxies = list(itertools.chain.from_iterable( \
                [secs[r[0]:r[1]:r[2]] for r in self.posetPEs]
            ))
        # print(feederProxies)

        feeders = sorted(self.posetPElist)
        hashes = sorted(self.hashPElist)
        overlapPElist = {}
        feederIdx = 0
        hashIdx = 0
        # Everything is sorted by PE, so we can proceed sequentially
        while feederIdx < len(feeders) and hashIdx < len(hashes):
            if feeders[feederIdx] == hashes[hashIdx]:
                overlapPElist[feeders[feederIdx]] = (feeders[feederIdx], hashes[hashIdx])
                feederIdx += 1
            hashIdx += 1
        self.overlapPElist = copy(overlapPElist)


        self.hWorkersFull = Group(HashWorker,args=[self.nodeConstructor, self.localVarGroup, self.thisProxy, self.hashPElist, self.posetPElist, overlapPElist])
        charm.awaitCreation(self.hWorkersFull)
        secs = [self.hWorkersFull[r[0]:r[1]:r[2]] for r in self.hashPEs]
        self.hWorkers = charm.combine(*secs)
        self.hashWorkerProxies = self.hWorkersFull.getProxies(ret=True).get()
        self.hashWorkerProxies = list(itertools.chain.from_iterable( \
                [self.hashWorkerProxies[r[0]:r[1]:r[2]] for r in self.hashPEs]
            ))
        self.amFeeder = False
        self.hashStoreMode = 0
        # self.hashWorkerProxies = self.hashWorkerProxies.get()

        self.enumChannels = {}

        if feederSpec is None:
            feederSpec = []
        feederKeys = ['nodeConstructor', 'localVarGroup', 'hashPEs', 'usePosetChecking', 'opts']
        if len(feederSpec) > 0:
            feederDef = feederSpec[0]
            assert isinstance(feederDef,dict)
            for ky in feederKeys:
                assert ky in feederDef
        # @coro
        # def initAsFeeder(self, nodeConstructor, localVarGroup, hashPEs, usePosetChecking, opts={} ):
            self.usePosetChecking = feederDef['usePosetChecking']
            self.nodeConstructorAsFeeder = feederDef['nodeConstructor']
            self.localVarGroupAsFeeder = feederDef['localVarGroup']
            self.targetHashPEs = feederDef['hashPEs']
            self.targetHashPElist = list(itertools.chain.from_iterable( \
                [list(range(r[0],r[1],r[2])) for r in self.targetHashPEs] \
                ))
            self.targetDistHashTable = Chare(DistHash,args=[ \
                self.hWorkersFull, \
                self.nodeConstructorAsFeeder, \
                self.localVarGroupAsFeeder , \
                self.targetHashPEs, \
                self.hashPEs, \
                feederSpec[1:] \
            ],onPE=0)
            charm.awaitCreation(self.targetDistHashTable)
            # self.targetDistHashTable.migrate(self.targetHashPElist[0],awaitable=True).get()
            self.hWorkersFull.setConstraint(**feederDef['opts'],awaitable=True).get()

            self.amFeeder = True
    @coro
    def getMigrationInfo(self):
        if self.amFeeder:
            # self.targetDistHashTable.migrate(self.targetHashPElist[0],awaitable=True).get()
            return [(self.targetHashPElist, self.targetDistHashTable)] + self.targetDistHashTable.getMigrationInfo(ret=True).get()
        else:
            return []
    @coro
    def getTargetDistHashProxy(self):
        return self.targetDistHashTable
    @coro
    def initialize(self):
        # self.hashWorkerChannels = [Channel(self, remote=proxy) for proxy in self.hashWorkerProxies]
        if self.amFeeder:
            # print('Initialized distHashTable group')
            initFut = self.targetDistHashTable.initialize(awaitable=True)
            initFut.get()
            if self.usePosetChecking:
                self.localVarGroupAsFeeder.init(self.hWorkersFull,self.hashPElist,awaitable=True).get()
        # Establish a feedback channel so that the hash table can send messages to the feeder workers:
        feeders = sorted(zip(self.posetPElist,self.feederProxies))
        hashes = sorted(zip(self.hashPElist,self.hashWorkerProxies))
        # TO DO
        # The presence of the following two lines seem to be a bug, since they ensure that no "overlap" PEs are detected.
        # As a result, no query mutex listener is started, so there is effectively no deadlock prevention when two successor worker
        # PEs suspend to query each other. These deadlocks were definitely occurring in pre-HSCC testing, but they don't seem to be
        # occurring now. I don't know it this is because I fixed a bug in hashNode (before many, many nodes were hashed to the same
        # value), and the deadlock is just much, much less likely to occur, or whether the query mutex is now no longer needed.
        # This needs to be ascertained, though.
        self.overlapPElist = {}
        self.mappedPElist = {}
        for idx in self.overlapPElist:
            self.overlapPElist[idx] = (feeders[self.overlapPElist[idx][0]][1], hashes[self.overlapPElist[idx][1]][1])


        myFut = self.feederGroup.initRateChannel(self.overlapPElist, awaitable=True)
        myFut.get()
        #self.feedbackChannels = [Channel(self, remote=proxy) for proxy in feederProxies]

        myFut = self.hWorkersFull.initRateChannelHashEnd(self.overlapPElist,awaitable=True)
        myFut.get()

        # Establish channels from each feeder worker to each hash worker
        myFut = self.feederGroup.initHashChannel(self.hashWorkerProxies , awaitable=True)
        myFut.get()

        myFut = self.hWorkersFull.initHashChannelHashEnd(self.feederProxies,awaitable=True)
        myFut.get()

        # Establish Query channels from each feeder worker to each hash worker
        myFut = self.feederGroup.initQueryChannel(self.hashWorkerProxies , self.thisProxy, awaitable=True)
        myFut.get()

        self.queryMutexChannelsHashEnd = []
        self.queryMutexChannelsHashEnd =  [ Channel(self, remote=self.overlapPElist[pxyIdx][0]) for pxyIdx in self.overlapPElist ]
        self.queryMutexLoopback = Channel(self, remote=self.thisProxy)
        self.queryMutexStatus = {}
        self.queryMutexFuts = {}
        for ch in self.queryMutexChannelsHashEnd:
            self.queryMutexStatus[ch] = 0
            self.queryMutexFuts[ch] = None
        self.queryMutexDone = None

        myFut = self.hWorkersFull.initQueryChannelHashEnd(self.feederProxies,awaitable=True)
        myFut.get()

    @coro
    def updateNodeEqualityFn(self,fn=None, nodeType='standard', tol=1e-9, rTol=1e-9, H=None):
        self.hWorkersFull.updateNodeEqualityFn(fn=fn,nodeType=nodeType,tol=tol,rTol=rTol, H=H, awaitable=True).get()

    @coro
    def decHashedNodeCountFeeder(self,pe):
        self.feederGroup[pe].decHashedNodeCount()

    @coro
    def queryMutexLocalListener(self,ch,chIdx):
        # print('Starting queryMutex LocalListener')
        while True:
            val = ch.recv()
            if val == -2:
                self.queryMutexStatus[ch] = -2
                val = -chIdx
                if any([self.queryMutexStatus[chIt] != -2 for chIt in self.queryMutexChannelsHashEnd]):
                    break
            else:
                val = chIdx

            # print('Query mutex requested for PE ' + str(val))
            ackFut = Future()
            self.queryMutexFuts[ch] = ackFut
            self.queryMutexLoopback.send(val)
            ackFut.get()
            if val < 0:
                break


    @coro
    def queryMutexListen(self):
        while True:
            chIdx = self.queryMutexLoopback.recv()
            # print('reqPE = ' + str(chIdx))
            if chIdx < 0:
                self.queryMutexFuts[self.queryMutexChannelsHashEnd[-chIdx]].send(1)
                self.queryMutexDone.send(1)
                break
            ch = self.queryMutexChannelsHashEnd[chIdx]
            ch.send(1)
            ch.recv()
            localFut = self.queryMutexFuts[ch]
            self.queryMutexFuts[ch] = None
            localFut.send(1)

    @coro
    def sendFeedbackMessage(self,msg):
        for ch in self.feedbackChannels:
            ch.send(msg)


    # This method is superceded by the DistributedHash.initListening -> DistributedHash.levelDone().get() sequence
    # It should be replaced by a method to signal the feeders on the feedback channel when early termination happens
    # @coro
    # def levelDoneChannel(self, doneFut):
    #     hashWorkerStatus = {}
    #     for ch in self.hashWorkerChannels:
    #         hashWorkerStatus[ch] = 0
    #     for ch in charm.iwait(self.hashWorkerChannels):
    #         val = ch.recv()
    #         if val == -2 or val == -3:
    #             hashWorkerStatus[ch] = val
    #         if all([hashWorkerStatus[ch] < 0 for ch in self.hashWorkerChannels]):
    #             doneFut.send(1)
    #             break

    @coro
    def levelDone(self):
        res = all(self.hWorkers.awaitLevel(ret=True).get())
        return res
    @coro
    def levelDoneSecondary(self):
        checkVal = None
        if self.amFeeder:
            if self.usePosetChecking:
                self.targetDistHashTable.awaitPending(awaitable=True).get()
            # print(f'Secondary Hash table shut down listener')
            self.hWorkers.sendAll(-2,awaitable=True).get()
            self.hWorkers.closeQueryChannels(awaitable=True).get()
            self.hWorkers.flushMessages(ret=True).get()

            # print('Finished looking for successors on level ' + str(level))
            checkVal = self.targetDistHashTable.levelDone(ret=True).get()
            # if not checkVal or timedOut:
            #     if timedOut: checkVal = None
        return None
    @coro
    def awaitShutdownSecondary(self):
        feederVal = None
        if self.amFeeder:
            feederVal = self.targetDistHashTable.awaitShutdown(ret=True).get()
        return feederVal
    @coro
    def getLevelList(self):
        nextlevel = Future()
        self.hWorkersFull.getLevelList(nextlevel)
        return nextlevel.get()

    @coro
    def scheduleNextLevel(self,clearTable=True):
        doneFuts = [Future() for k in range(len(self.feederProxies))]
        for k in range(len(doneFuts)):
            self.feederProxies[k].initList(doneFuts[k])

        cnt = 0
        for fut in charm.iwait(doneFuts):
            cnt += fut.get()

        nextLevel = self.hWorkersFull.getLevelSizes(ret=True).get()
        nextLevelSize = sum(nextLevel)

        hashPEidx = 0
        doneFuts = []
        feederPEOffset = 0
        for hashPEidx in range(len(self.hashPElist)):
            if nextLevel[self.hashPElist[hashPEidx]] > 0:
                doneFuts.append(Future())
                self.hashWorkerProxies[hashPEidx].distributeTableChunk(feederPEOffset,doneFuts[-1],clearTable=clearTable)
                feederPEOffset = nextLevel[self.hashPElist[hashPEidx]] % len(self.posetPElist)
        cnt = 0
        for fut in charm.iwait(doneFuts):
            cnt += fut.get()

        return nextLevelSize




    @coro
    def awaitPending(self, usePosetChecking=True):
        outstandingHashesPrev = None
        cnt = 0
        while usePosetChecking:
            outstandingHashes = self.feederGroup.getHashedNodeCount(ret=True).get()
            hashCount = [ x == 0 for x in outstandingHashes ]
            # print(f'hashCount = {hashCount} P{self.feederGroup.getHashedNodeCount(ret=True).get()}')
            # print(f'PE {charm.myPe()}: pendingCnt = {(hashCount, schedCount)}; hashPEs = {self.hashPElist}')
            if all(hashCount):
                break
            if not outstandingHashesPrev is None and sum(outstandingHashes) >= sum(outstandingHashesPrev):
                if cnt < 9:
                    f = Future()
                    f.send(1)
                    f.get()
                else:
                    charm.sleep(0.001)
            outstandingHashesPrev = outstandingHashes
            cnt = (cnt + 1) % 10
        if usePosetChecking:
            self.feederGroup.resetHashedNodeCount(awaitable=True).get()

    @coro
    def awaitShutdown(self):
        if not self.queryMutexDone is None:
            self.queryMutexDone.get()
        all(self.hWorkers.awaitQueries(ret=True).get())
        listenerShutdownFut = Future()
        # print('Awaiting listener shutdown...')
        self.hWorkersFull.awaitListenerShutdown(listenerShutdownFut)
        val = listenerShutdownFut.get()
        # print('Finished listener shutdown...')
        # print('Count was ' + str(val))
        return val

    @coro
    def getWorkerProxy(self):
        return self.hWorkers
    @coro
    def getWorkerProxyFull(self):
        return self.hWorkersFull
    @coro
    def initListening(self,allDone,queryReturnInfo=False):
        for ch in self.queryMutexChannelsHashEnd:
            self.queryMutexStatus[ch] = 0
        if len(self.queryMutexChannelsHashEnd) > 0:
            for k in range(len(self.queryMutexChannelsHashEnd)):
                self.thisProxy.queryMutexLocalListener(self.queryMutexChannelsHashEnd[k],k)
            self.queryMutexDone = Future()
            self.thisProxy.queryMutexListen()
        doneFuts = [Future() for k in range(len(self.hashWorkerProxies))]
        for k in range(len(doneFuts)):
            self.hashWorkerProxies[k].initListen(doneFuts[k],queryReturnInfo=queryReturnInfo)
        cnt = 0
        for fut in charm.iwait(doneFuts):
            cnt += fut.get()
        # print('**** Count is ' + str(cnt) + ' ****')


        allDone.send(cnt)
        self.hWorkers.listen()
        # print('All workers done!')
        # self.hWorkers.listen()

    @coro
    def initListeningSecondary(self,allDone):
        self.targetDistHashTable.initListening(allDone,awaitable=True).get()

    @coro
    def startListeningSecondary(self):
        self.hWorkersFull.startListening(awaitable=True).get()

    @coro
    def clearHashTable(self):
        self.hWorkersFull.clearHashTable(awaitable=True).get()

    @coro
    def resetLevelCount(self):
        self.hWorkersFull.resetLevelCount(awaitable=True).get()

    @coro
    def getTable(self):
        return self.hWorkersFull.getTable(ret=True).get()

    @coro
    def getTableLen(self):
        return sum(self.hWorkersFull.getTableLen(ret=True).get())
    @coro
    def _NOOP_(self):
        pass
    @coro
    def registerEnumChannels(self, remChare):
        if remChare in self.enumChannels:
            print(f'Warning: remote Chare already registered!')
            return
        chans = {'extData':Channel(self,remote=remChare), 'extCtrl':Channel(self,remote=remChare), 'hashWorkers':[]}
        remChare.registerEnum(self.thisProxy,awaitable=True).get()
        chans['hashWorkers'] = [{'data':Channel(self,remote=r),'ctrl':Channel(self,remote=r)} for r in self.hashWorkerProxies]
        for r in self.hashWorkerProxies:
            r.registerEnumHashEnd(self.thisProxy,remChare,awaitable=True).get()
        self.enumChannels[remChare] = chans

    @coro
    def enumTable(self, remChare, statusFut):
        if not remChare in self.enumChannels:
            print(f'Error: remote chare is not registered!')
            return
        dataChan = self.enumChannels[remChare]['extData']
        ctrlChan = self.enumChannels[remChare]['extCtrl']
        hashChannels = self.enumChannels[remChare]['hashWorkers']

        listenerStats = []
        remsReady = []
        for r in self.hashWorkerProxies:
            f = Future()
            r.enumListener(remChare,f)
            listenerStats.append(f)
        for remStat in charm.iwait(listenerStats):
            v = remStat.get()
            if v >= 0:
                remsReady.append(v)
        if len(remsReady) < len(self.hashWorkerProxies):
            for rIdx in remsReady:
                hashChannels[rIdx]['ctrl'].send(-1)
                rcvVal = hashChannels[r]['data'].recv()
                while not rcvVal is None:
                    rcvVal = hashChannels[rIdx]['data'].recv()
            print(f'Failed to start all enumerator listeners on hash workers. Exiting...')
            statusFut.send(False)
            return
        statusFut.send(True)
        print(f'Successfully started all enumerator listeners')
        term = False
        next = False
        for rIdx in range(len(self.hashWorkerProxies)):
            ch = hashChannels[rIdx]
            dataVal = 1
            while not dataVal is None:
                if next and not term:
                    ctrlVal = 1
                    next = False
                elif term:
                    ctrlVal = -1
                else:
                    ctrlVal = ctrlChan.recv()
                if ctrlVal > 0:
                    # print(f'Passing next signal to hash workers')
                    ch['ctrl'].send(1)
                else:
                    ch['ctrl'].send(-1)
                    dataVal = ch['data'].recv()
                    # print(f'Shutting down enumListener on PE {charm.myPe()} (received {dataVal})')
                    assert dataVal is None
                    term = True
                    continue
                dataVal = ch['data'].recv()
                if not dataVal is None:
                    dataChan.send(dataVal)
                else:
                    # print(f'Sent transition value on control channel')
                    if rIdx < len(self.hashWorkerProxies) - 1:
                        next = True
        dataChan.send(None)




        # for ch in hashChannels:
        #     ch['ctrl'].send(-1)
        # for ch in hashChannels:
        #     rcvVal = ch['data'].recv()
        #     while not rcvVal is None:
        #         rcvVal = ch['data'].recv()
        #     print(f'Finished shutting down enumerator listener {ch}')



