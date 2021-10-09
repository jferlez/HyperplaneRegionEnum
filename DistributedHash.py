import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
from numpy import may_share_memory
import time
import itertools
import random



class Node():

    def __init__(self,localProxy, parentChare, lsb,msb,nodeInt, *args):
        self.lsbHash = lsb
        self.msbHash = msb
        self.nodeInt = nodeInt
        self.localProxy = localProxy
        self.parentChare = parentChare
        self.data = args
    
    def __hash__(self):
        return self.msbHash
    
    def __eq__(self,other):
        if type(other) == int:
            return self.nodeInt == other
        elif isinstance(other,Node):
            return self.nodeInt == other.nodeInt



def Join(contribs):
    return list(itertools.chain.from_iterable(contribs))
Reducer.addReducer(Join)

class HashWorker(Chare):

    def __init__(self,nodeConstructor,localVarGroup,parentProxy,pes):
        self.hashPElist = pes
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
        #print(self.thisIndex)
    
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
        for ch in self.inChannels:
            self.status[ch] = 0
            self.messages[ch] = {'msg':None, 'fut':None}
            self.workerDone[ch] = None
    

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
        for ch in self.queryChannels:
            self.queryStatus[ch] = 0
            self.queryMessages[ch] = {'msg':None, 'fut':None}
            self.queryDone[ch] = None
    
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
        while True:
            # print('Listener started')
            # charm.wait([ch])
            # sig = str(random.random())
            # before = [self.messages[chIt]['msg'] for chIt in self.inChannels]
            val = ch.recv()
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

    @coro
    def initListen(self,fut):
        if not charm.myPe() in self.hashPElist:
            return
        for ch in self.inChannels:
            self.status[ch] = 0
            self.messages[ch] = {'msg':None, 'fut':None}
            self.workerDone[ch] = None
        for k in range(len(self.inChannels)):
            self.thisProxy[self.thisIndex].localListener(self.inChannels[k],k)
            self.workerDone[self.inChannels[k]] = Future()
        
        for ch in self.queryChannels:
            self.queryStatus[ch] = 0
            self.queryMessages[ch] = {'msg':None, 'fut':None}
            self.queryDone[ch] = None
        for k in range(len(self.queryChannels)):
            self.thisProxy[self.thisIndex].localQueryListener(self.queryChannels[k],k)
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
                if all([self.status[ch] <= -2 for ch in self.inChannels]) and \
                        all([self.queryStatus[ch] <= -2 for ch in self.queryChannels]):
                    self.rateChannel.send(min([self.status[ch] for ch in self.inChannels]))
                    break
                # print('Received control of ' + str(control) + ' on PE ' + str(charm.myPe()))
                if all([self.messages[ch]['fut'] is None for ch in self.inChannels]) and \
                    all([self.queryMessages[ch]['fut'] is None for ch in self.queryChannels]):
                    self.rateChannel.send(-1)
                    continue
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
                        if charm.myPe() != self.hashPElist[chIdx] or (not selfQuery or charm.myPe() == self.hashPElist[chIdx]):
                            answeredSelf = True
                        newNode = self.nodeConstructor(self.localVarGroup[charm.myPe()], self, *val)
                        if newNode in self.table:
                            # print('Responding to query ' + str(val) + ' on channel ' + str(chIdx))
                            self.queryChannels[chIdx].send(val[2])
                        else:
                            # print('Responding to query ' + str(val) + ' on channel ' + str(chIdx))
                            self.queryChannels[chIdx].send(-val[2])
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
                        for ch in self.inChannels:
                            if self.status[ch] != -2 and self.status[ch] != -3:
                                self.workerDone[ch].send(True)
                            self.status[ch] = -3
                    elif val == -2:
                        if self.status[ch] != -2 and self.status[ch] != -3:
                                self.workerDone[ch].send(True)
                                self.status[ch] = -2
                        # self.workerDone[ch].send(True)
                        self.levelDone = True
                    elif type(val) == tuple and len(val) >= 3:
                        newNode = self.nodeConstructor(self.localVarGroup[charm.myPe()], self, *val)
                        if self.nodeCalls & 1:
                            newNode.init()
                        if not newNode in self.table:
                            self.table[newNode] = {'nodeInt': val[2], 'checked':False}
                            self.levelList.append(val[2])
                            # Check node here:
                            if False: # If result of node check is False return False on all the workerDone Futures
                                if self.status[ch] != -2 and self.status[ch] != -3 and not self.workerDone[ch] is None:
                                    self.workerDone[ch].send(False)
                                self.status[ch] = -3
                                self.parentProxy.sendFeedbackMessage(charm.numPes()+1)
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
        retVal = all([self.workerDone[ch].get() for ch in self.inChannels]) and \
            all([self.queryDone[ch].get() for ch in self.queryChannels])
        self.levelDone = True
        return retVal

    @coro
    def getLevelList(self, levelListFut):
        if self.levelDone:
            self.reduce(levelListFut, self.levelList, Reducer.Join)
        else:
            print('Warning: tried to retrieve level list before level was done!')
            self.reduce(levelListFut, [], Reducer.Join)
    # @coro
    # def receiveOnNodeChannel(self):
    #     if not charm.myPe() in self.hashPElist:
    #         return
    #     count = 0
    #     for i in charm.iwait([self.inChannels[ch].recv() for ch in self.inChannels]):
    #         count += 1
    #     return count
    

class DistHash(Chare):

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
        self.hWorkersFull = Group(HashWorker,args=[self.nodeConstructor, self.localVarGroup, self.thisProxy, self.hashPElist])
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
        # Get a list of proxies for all memembers of the feeder group:
        secs = self.feederGroup.getProxies(ret=True).get()
        feederProxies = list(itertools.chain.from_iterable( \
                [secs[r[0]:r[1]:r[2]] for r in self.posetPEs]
            ))
        # print(feederProxies)
        # Establish a feedback channel so that the hash table can send messages to the feeder workers:
        feeders = sorted(zip(self.posetPElist,feederProxies))
        hashes = sorted(zip(self.hashPElist,self.hashWorkerProxies))
        self.overlapPElist = {}
        self.mappedPElist = {}
        feederIdx = 0
        hashIdx = 0
        # Everything is sorted by PE, so we can proceed sequentially
        while feederIdx < len(feeders) and hashIdx < len(hashes):
            if feeders[feederIdx][0] == hashes[hashIdx][0]:
                self.overlapPElist[feeders[feederIdx][0]] = (feeders[feederIdx][1], hashes[hashIdx][1])
                feederIdx += 1
            else:
                if feeders[feederIdx][0] in self.mappedPElist:
                    self.mappedPElist[hashes[hashIdx][0]].append((feeders[feederIdx][1], hashes[hashIdx][1]))
                else:
                    self.mappedPElist[hashes[hashIdx][0]] = [(feeders[feederIdx][1], hashes[hashIdx][1])]
            hashIdx += 1

        myFut = self.feederGroup.addFeedbackRateChannelOrigin(self.overlapPElist, awaitable=True)
        myFut.get()        
        #self.feedbackChannels = [Channel(self, remote=proxy) for proxy in feederProxies]

        myFut = self.hWorkersFull.addFeedbackRateChannelDest(self.overlapPElist,awaitable=True)
        myFut.get()
        
        # Establish channels from each feeder worker to each hash worker
        myFut = self.feederGroup.addDestChannel(self.hashWorkerProxies , awaitable=True)
        myFut.get()

        myFut = self.hWorkersFull.addOriginChannel(feederProxies,awaitable=True)
        myFut.get()

        # Force the node channels to bind first, so that we can add query channels
        # self.feederGroup.sendAll(-2,awaitable=True).get()
        # self.hWorkersFull.receiveOnNodeChannel(ret=True).get()

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

        myFut = self.hWorkersFull.addQueryOriginChannel(feederProxies,awaitable=True)
        myFut.get()

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
        self.queryMutexDone.get()
        return nextlevel.get()

    def getWorkerProxy(self):
        return self.hWorkers
    @coro
    def initListening(self,allDone):
        for ch in self.queryMutexChannels:
            self.queryMutexStatus[ch] = 0
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