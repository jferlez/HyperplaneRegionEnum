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
from numba.pycc import CC
from functools import partial

XFER_CHUNK_SIZE = 1000

class Node():

    def __init__(self,localProxy, storePe, parentChare, nodeEqualityFn, lsb,msb,nodeBytes, originPe, *args):
        self.lsbHash = lsb
        self.msbHash = msb
        self.nodeBytes = nodeBytes
        self.localProxy = localProxy
        self.storePe = storePe
        self.parentChare = parentChare
        self.originPe = originPe
        self.nodeEqualityFn = nodeEqualityFn
        self.payload = args
    
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
        self.inChannels = []
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
        self.parentChannel = Channel(self,remote=self.parentProxy)
        self.loopback = Channel(self,remote=self.thisProxy[self.thisIndex])
        self.queryLoopback = Channel(self,remote=self.thisProxy[self.thisIndex])
        self.controlLoopback = Channel(self,remote=self.thisProxy[self.thisIndex])
        self.nodeEqualityFn = lambda x,y: (x == y)
        #print(self.thisIndex)
    
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
    def addOriginChannel(self,feederProxies):
        if not charm.myPe() in self.hashPElist:
            return
        self.feederProxies = feederProxies
        self.numFeederWorkers = len(feederProxies)
        self.inChannels = [Channel(self, remote=proxy) for proxy in feederProxies]
        self.status = {}
        self.messages = {}
        self.workerDone = {}
        self.localListenerOnline = {}
        for ch in self.inChannels:
            self.status[ch] = 0
            self.messages[ch] = {'msg':None, 'fut':None}
            self.workerDone[ch] = None
            self.localListenerOnline[ch] = False
    

    @coro
    def addQueryOriginChannel(self,feederProxies):
        if not charm.myPe() in self.hashPElist:
            return
        # Call this **after** establishing the hashing channels
        # self.feederProxies = feederProxies
        # self.numFeederWorkers = len(feederProxies)
        self.queryChannels = [Channel(self, remote=proxy) for proxy in feederProxies]
        self.queryStatus = {}
        self.queryMessages = {}
        self.queryDone = {}
        self.queryListenerOnline = {}
        for ch in self.queryChannels:
            self.queryStatus[ch] = 0
            self.queryMessages[ch] = {'msg':None, 'fut':None}
            self.queryDone[ch] = None
            self.queryListenerOnline[ch] = False
    
    # The following two methods allow a DistributedHash to function as a "feeder" to **another** DistributedHash!
    # To do: implement the feedback channel from the other DistributedHash...
    @coro
    def addDestChannel(self, procGroupProxies):
        if not charm.myPe() in self.hashPElist:
            return
        self.numHashWorkers = len(procGroupProxies)
        self.outChannels = [Channel(self, remote=proxy) for proxy in procGroupProxies]
        self.numHashBits = 1
        while self.numHashBits < self.numHashWorkers:
            self.numHashBits = self.numHashBits << 1
        self.hashMask = self.numHashBits - 1
        self.numHashBits -= 1
        if self.N % 4 == 0:
            self.numBytes = self.N/4
        else:
            self.numBytes = int(self.N/4)+1
        # print(self.outChannels)
    

    @coro
    def addFeedbackChannel(self,proxy):
        if not charm.myPe() in self.hashPElist:
            return
        self.feedbackChannel = Channel(self,remote=proxy)
    @coro
    def addFeedbackRateChannelDest(self,overlapPElist):
        self.rateChannel = None
        self.termProxy = None
        self.overlapPElist = overlapPElist
        if not charm.myPe() in self.hashPElist:
            return
        if charm.myPe() in overlapPElist:
            self.rateChannel = Channel(self,remote=overlapPElist[charm.myPe()][0])
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
            # before = [self.messages[chIt]['msg'] for chIt in self.inChannels]
            val = ch.recv()
            # print('Recieved val ' + str(val) + 'on PE ' + str(charm.myPe()))
            # Make sure we don't start reading until the previous level/poset was done
            if not validInput:
                if val == -100:
                    validInput = True
                continue
            # print(val)
            self.messages[ch]['msg'] = val
            ackFut = Future()
            self.messages[ch]['fut'] = ackFut    
            self.loopback.send(chIdx)
            # if not self.initiatedNodeProc:
            #     print('----'*(charm.myPe()+1) + '>>  PE'+str(charm.myPe())+'LocalListener ' + sig + ' -- Signaling to listen()' )
            if not self.initiatedNodeProc:
                self.initiatedNodeProc = True
                self.controlLoopback.send(1)
            # self.parentProxy.sendFeedbackMessage(-1*charm.myPe())
            # print('----'*(charm.myPe()+1) + '>>  PE'+str(charm.myPe())+'LocalListener ' + sig + ' -- initiatedNodeProc = ' + str(self.initiatedNodeProc) + ' Before: ' + str(before) + ' After: ' + str([self.messages[chIt]['msg'] for chIt in self.inChannels]))
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
            # before = [self.queryMessages[chIt]['msg'] for chIt in self.queryChannels]
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
            # print('===='*(charm.myPe()+1) + '>>  PE'+str(charm.myPe())+'QUERYLocalListener ' + sig + ' -- initiatedQueryProc = ' + str(self.initiatedNodeProc) + ' Before: ' + str(before) + ' After: ' + str([self.queryMessages[chIt]['msg'] for chIt in self.queryChannels]))
            ackFut.get()
            # self.parentProxy.sendFeedbackMessage(charm.myPe())
            self.queryMessages[ch]['msg'] = None
            self.queryMessages[ch]['fut'] = None
            if val == -2:
                # print('===='*(charm.myPe()+1) + '>>  PE'+str(charm.myPe())+'QUERYLocalListener ' + sig + ' -- TERMINATING!')
                break
        return 1

    @coro
    def initListen(self,fut):
        if not charm.myPe() in self.hashPElist:
            return
        # Make sure all of the loopback channels are empty:
        for loopbackIt in [self.loopback, self.queryLoopback, self.controlLoopback]:
            loopbackIt.send(-100)
            while True:
                m = loopbackIt.recv()
                if m == -100:
                    break
        for ch in self.inChannels:
            self.status[ch] = 0
            self.messages[ch] = {'msg':None, 'fut':None}
            self.workerDone[ch] = None
        self.listenerStatus = []
        for k in range(len(self.inChannels)):
            self.listenerStatus.append( self.thisProxy[self.thisIndex].localListener(self.inChannels[k],k, ret=True) )
            self.workerDone[self.inChannels[k]] = Future()
        
        for ch in self.queryChannels:
            self.queryStatus[ch] = 0
            self.queryMessages[ch] = {'msg':None, 'fut':None}
            self.queryDone[ch] = None
        for k in range(len(self.queryChannels)):
            self.listenerStatus.append( self.thisProxy[self.thisIndex].localQueryListener(self.queryChannels[k],k,ret=True) )
            self.queryDone[self.queryChannels[k]] = Future()
        
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
        for ch in self.inChannels:
            msgCount[ch] = 0
        while any([self.queryStatus[ch] > -2 for ch in self.queryChannels]) or \
            any([self.status[ch] > -2 for ch in self.inChannels]): # or not free:
            # traceSig = random.random()
            # peSig = 2*charm.myPe()*'    ' + 'PE_' + str(charm.myPe()) + 2*(charm.numPes()+4 - charm.myPe())*' ' + '  '
            # indent = cnt*8*' '
            # prefix = peSig + indent + str(traceSig)
            # print(prefix + ' Hash worker on PE ' + str(charm.myPe()) + ' START STATE ---> [pendingChecks, pendingQueries] = ' + str([pendingChecks, pendingQueries]) +  \
            #     'Node Listener status: ' + str([self.status[chIt] for chIt in self.inChannels]) + ' Query Listener status: ' + str([self.queryStatus[chIt] for chIt in self.queryChannels]) + ' ' \
            #         + 'Node messages buffered: ' + str([self.messages[ch]['msg'] for ch in self.inChannels]) + ' Query messages buffered: ' + str([self.queryMessages[ch]['msg'] for ch in self.queryChannels])  + ' ' \
            #             + 'Message buffers: ' + str([self.messages[ch]['fut'] for ch in self.inChannels]) + str([self.queryMessages[ch]['fut'] for ch in self.queryChannels]))
            # If we're running on the same PE as a feeder worker, and he hasn't signaled "free run mode" (i.e. he has no more work to do)
            # then wait for the feeder to transfer control
            if not self.rateChannel is None and not free:
                control = self.rateChannel.recv()
                # print(prefix + ' RECEIVED RELEASE CONTROL SIGNAL ' + str(control) + ' on PE ' + str(charm.myPe()) + ' ')
                if control == 2:
                    free = True
                    queryOnly = False
                    selfQuery = False
                    processOnly = True 
                if control <= 0:
                    self.rateChannel.send(control)
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
                if all([self.status[ch] <= -2 for ch in self.inChannels]) and \
                        all([self.queryStatus[ch] <= -2 for ch in self.queryChannels]):
                    self.rateChannel.send(min([self.status[ch] for ch in self.inChannels]))
                    break
                # print('Received control of ' + str(control) + ' on PE ' + str(charm.myPe()))
                if all([self.messages[ch]['fut'] is None for ch in self.inChannels]) and \
                    all([self.queryMessages[ch]['fut'] is None for ch in self.queryChannels]):
                    self.rateChannel.send(-1)
                    continue
                if queryOnly and not selfQuery and all([self.queryMessages[ch]['fut'] is None for ch in self.queryChannels]):
                    self.rateChannel.send(-1)
                    continue
            
            # print(prefix + ' Starting trace on PE ' + str(charm.myPe()) + ' -------------------------')
            pendingWork = [3,3]
            if not processOnly:    
                pendingWork[0] = self.controlLoopback.recv()
                if ((not pendingChecks and pendingWork[0] > 1) and any([not self.messages[ch]['fut'] is None for ch in self.inChannels])) or \
                        pendingWork[0] == 1 and any([not self.queryMessages[ch]['fut'] is None for ch in self.queryChannels]):
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
            #     'Node Listener status: ' + str([self.status[chIt] for chIt in self.inChannels]) + ' Query Listener status: ' + str([self.queryStatus[chIt] for chIt in self.queryChannels]) + ' ' \
            #         + 'Node messages buffered: ' + str([self.messages[ch]['msg'] for ch in self.inChannels]) + ' Query messages buffered: ' + str([self.queryMessages[ch]['msg'] for ch in self.queryChannels])  + ' ' \
            #             + 'Message buffers: ' + str([self.messages[ch]['fut'] for ch in self.inChannels]) + str([self.queryMessages[ch]['fut'] for ch in self.queryChannels]))
                
            
            # Count the number of times we execute when there are queries pending:
            if pendingChecks or not pendingQueries or all([self.queryStatus[chIt] == -2 for chIt in self.queryChannels]):
                yieldCount += 1
            
            # print(prefix + ' Hash worker on PE ' + str(charm.myPe()) + ' pendingWork=' + str(pendingWork) + ' ' + \
            #    str([self.messages[ch]['msg'] for ch in self.inChannels]) + ' ' + \
            #        str([self.queryMessages[ch]['msg'] for ch in self.queryChannels])  + ' ' + str(self.status))
            # Respond to hash table queries
            chList = []
            answeredSelf = False
            if pendingQueries:
                numPending = sum([not self.queryMessages[ch]['fut'] is None for ch in self.queryChannels])
                # print('numPending on PE ' + str(charm.myPe()) + ' is ' + str(numPending))
                i = 0
                while i < numPending:
                    chIdx = self.queryLoopback.recv()
                    ch = self.queryChannels[chIdx]
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
                            self.queryChannels[chIdx].send(1)
                        else:
                            # print('Responding to query ' + str(val) + ' on channel ' + str(chIdx))
                            self.queryChannels[chIdx].send(-1)
                    elif val < 0:
                        answeredSelf = True
                        # for chIt in self.queryChannels:
                        self.queryStatus[ch] = -2
                        self.queryDone[ch].send(True)
                    else:
                        print('PE' + str(charm.myPe()) + ' : Unexpected query received. Query was ' + str(val))
                    i += 1
                    # Ensure we catch any messages arriving at the local listener while we were receiving on the queryLoopback channel
                    numPending = sum([not self.queryMessages[ch]['fut'] is None for ch in self.queryChannels])
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
                numPending = sum([not self.messages[ch]['fut'] is None for ch in self.inChannels])
                # print(prefix + 'numPending nodes is ' + str(numPending) + 'on PE ' + str(charm.myPe()))
                i = 0
                while i < numPending:
                    chIdx = self.loopback.recv()
                    # print(prefix + ' Recieved a chIdx of ' + str(chIdx) + ' on PE ' + str(charm.myPe()))
                    ch = self.inChannels[chIdx]
                    msg = self.messages[ch]
                    # print(prefix + ' Processing node ' + str(msg) + ' on PE ' + str(charm.myPe()))
                    chList.append(ch)
                    val = msg['msg']
                    msgCount[ch] += 1
                    if val == -3:
                        # for ch in self.inChannels:
                        if self.status[ch] != -2 and self.status[ch] != -3:
                            self.workerDone[ch].send(True)
                        self.status[ch] = -3
                        self.localVarGroup.setSkip(True)
                    elif val == -4:
                        # msgFalseSet = False
                        # for ch in self.inChannels:
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
                        if all([self.status[chIt] <= -2 for chIt in self.inChannels]):
                            self.levelDone = True
                    elif type(val) == tuple and len(val) >= 3:
                        newNode = self.nodeConstructor(self.localVarGroup, charm.myPe(), self, self.nodeEqualityFn, *val)
                        if self.nodeCalls & 1:
                            newNode.init()
                        if not newNode in self.table:
                            self.table[newNode] = {'checked':False}
                            # self.levelList.append((val[2],*newNode.payload))
                            self.levelList.append(newNode)
                            # Check node here:
                            if self.nodeCalls & 4 and not newNode.check(): # If result of node check is False return False on all the workerDone Futures
                                if self.status[ch] != -2 and self.status[ch] != -3 and not self.workerDone[ch] is None:
                                    self.workerDone[ch].send(False)
                                self.status[ch] = -3
                                # self.parentProxy.sendFeedbackMessage(charm.numPes()+1)
                                self.levelDone = True
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
                    numPending = sum([not self.messages[ch]['fut'] is None for ch in self.inChannels])
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
            retControl = -3 if any([self.status[ch] == -3 for ch in self.inChannels]) else -1
            if not self.rateChannel is None and not free:
                self.rateChannel.send(retControl)
            elif not self.termProxy is None and retControl == -3:
                self.termProxy.sendAll(-3, awaitable=True).get()
            # print(prefix + ' Hash worker on PE ' + str(charm.myPe()) + ' FINAL STATE ---> pendingWork=' + str(pendingWork) + ' ' +  \
            #     'Node Listener status: ' + str([self.status[chIt] for chIt in self.inChannels]) + ' Query Listener status: ' + str([self.queryStatus[chIt] for chIt in self.queryChannels]) + ' ' \
            #         + 'Node messages buffered: ' + str([self.messages[ch]['msg'] for ch in self.inChannels]) + ' Query messages buffered: ' + str([self.queryMessages[ch]['msg'] for ch in self.queryChannels])  + ' ' \
            #             + 'Message buffers: ' + str([self.messages[ch]['fut'] for ch in self.inChannels]) + str([self.queryMessages[ch]['fut'] for ch in self.queryChannels]))
            # print(prefix + ' Finished trace on PE ' + str(charm.myPe()) + ' ----------------------')
            cnt += 1
        # print('Shutting down main listener on PE ' + str(charm.myPe()))
        return 1

    @coro
    def awaitLevel(self):
        retVal = all([self.workerDone[ch].get() for ch in self.inChannels])
        self.levelDone = True
        return retVal
    @coro
    def awaitQueries(self):
        return all([self.queryDone[ch].get() for ch in self.queryChannels])
        
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
    def clearHashTable(self):
        self.levelList = []
        self.table = {}
    @coro
    def resetLevelCount(self):
        self.level=-1
    # @coro
    # def receiveOnNodeChannel(self):
    #     if not charm.myPe() in self.hashPElist:
    #         return
    #     count = 0
    #     for i in charm.iwait([self.inChannels[ch].recv() for ch in self.inChannels]):
    #         count += 1
    #     return count
    @coro
    def distributeTableChunk(self,feederPEoffset,retFut,clearTable=True):
        while len(self.levelList) > 0:
            chunkSize = min(XFER_CHUNK_SIZE * len(self.feederPElist), len(self.levelList))
            xferDone = [Future() for _ in range(len(self.feederPElist))]
            for feederPEidx in range(len(self.feederPElist)):
                idx = (feederPEidx + feederPEoffset) % len(self.feederPElist)
                self.feederProxies[idx].appendToWorkList([(nd.nodeBytes, *nd.payload) for nd in self.levelList[feederPEidx:chunkSize:len(self.feederPElist)]],xferDone[feederPEidx])
            cnt = 0
            for fut in charm.iwait(xferDone):
                cnt += fut.get()
            if clearTable:
                for idx in range(chunkSize):
                    self.table.pop(self.levelList[idx])
            self.levelList = self.levelList[chunkSize:]
        retFut.send(1)
    

class DistHash(Chare):
    @coro
    def __init__(self, feederGroup, nodeConstructor, localVarGroup, hashPEs, posetPEs):
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
        # self.hashWorkerProxies = self.hashWorkerProxies.get()
        self.hashWorkerChannels = [Channel(self, remote=proxy) for proxy in self.hashWorkerProxies]

    @coro
    def initialize(self):

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


        myFut = self.feederGroup.addFeedbackRateChannelOrigin(self.overlapPElist, awaitable=True)
        myFut.get()        
        #self.feedbackChannels = [Channel(self, remote=proxy) for proxy in feederProxies]

        myFut = self.hWorkersFull.addFeedbackRateChannelDest(self.overlapPElist,awaitable=True)
        myFut.get()
        
        # Establish channels from each feeder worker to each hash worker
        myFut = self.feederGroup.addDestChannel(self.hashWorkerProxies , awaitable=True)
        myFut.get()

        myFut = self.hWorkersFull.addOriginChannel(self.feederProxies,awaitable=True)
        myFut.get()

        # Establish Query channels from each feeder worker to each hash worker
        myFut = self.feederGroup.addQueryDestChannel(self.hashWorkerProxies , self.thisProxy, awaitable=True)
        myFut.get()

        self.queryMutexChannels = []
        self.queryMutexChannels =  [ Channel(self, remote=self.overlapPElist[pxyIdx][0]) for pxyIdx in self.overlapPElist ]
        self.queryMutexLoopback = Channel(self, remote=self.thisProxy)
        self.queryMutexStatus = {}
        self.queryMutexFuts = {}
        for ch in self.queryMutexChannels:
            self.queryMutexStatus[ch] = 0
            self.queryMutexFuts[ch] = None
        self.queryMutexDone = None

        myFut = self.hWorkersFull.addQueryOriginChannel(self.feederProxies,awaitable=True)
        myFut.get()

    @coro
    def updateNodeEqualityFn(self,fn=None, nodeType='standard', tol=1e-9, rTol=1e-9, H=None):
        self.hWorkersFull.updateNodeEqualityFn(fn=fn,nodeType=nodeType,tol=tol,rTol=rTol, H=H, awaitable=True).get()

    @coro
    def queryMutexLocalListener(self,ch,chIdx):
        # print('Starting queryMutex LocalListener')
        while True:
            val = ch.recv()
            if val == -2:
                self.queryMutexStatus[ch] = -2
                val = -chIdx
                if any([self.queryMutexStatus[chIt] != -2 for chIt in self.queryMutexChannels]):
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
                self.queryMutexFuts[self.queryMutexChannels[-chIdx]].send(1)
                self.queryMutexDone.send(1)
                break
            ch = self.queryMutexChannels[chIdx]
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
        return all(self.hWorkers.awaitLevel(ret=True).get())
    
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
    def awaitPending(self):
        while True:
            pendingCnt = sum(self.localVarGroup.getSchedCount(ret=True).get())
            if pendingCnt == 0:
                break
    
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


    def getWorkerProxy(self):
        return self.hWorkers
    @coro
    def initListening(self,allDone):
        for ch in self.queryMutexChannels:
            self.queryMutexStatus[ch] = 0
        if len(self.queryMutexChannels) > 0:
            for k in range(len(self.queryMutexChannels)):
                self.thisProxy.queryMutexLocalListener(self.queryMutexChannels[k],k)
            self.queryMutexDone = Future()
            self.thisProxy.queryMutexListen()
        doneFuts = [Future() for k in range(len(self.hashWorkerProxies))]
        for k in range(len(doneFuts)):
            self.hashWorkerProxies[k].initListen(doneFuts[k])
        cnt = 0
        for fut in charm.iwait(doneFuts):
            cnt += fut.get()
        # print('**** Count is ' + str(cnt) + ' ****')
        
        
        allDone.send(cnt)
        self.hWorkers.listen()
        # print('All workers done!')
        # self.hWorkers.listen()
    
    @coro
    def clearHashTable(self):
        self.hWorkersFull.clearHashTable(awaitable=True).get()
    
    @coro
    def resetLevelCount(self):
        self.hWorkersFull.resetLevelCount(awaitable=True).get()