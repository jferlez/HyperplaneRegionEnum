import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
from numpy import may_share_memory
import time
import itertools



class Node():

    def __init__(self,lsb,msb,nodeInt,localProxy, parentChare):
        self.lsbHash = lsb
        self.msbHash = msb
        self.nodeInt = nodeInt
        self.localProxy = localProxy
        self.parentChare = parentChare
    
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
        self.numFeederWorkers = len(feederProxies)
        self.inChannels = [Channel(self, remote=proxy) for proxy in feederProxies]
        self.status = {}
        self.messages = {}
        self.workerDone = {}
        for ch in self.inChannels:
            self.status[ch] = 0
            self.messages[ch] = {'msg':None, 'fut':None}
            self.workerDone[ch] = None
    
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
    def localListener(self,ch,chIdx):
        while True:
            # print('Listener started')
            # charm.wait([ch])
            val = ch.recv()
            # print(val)
            self.messages[ch]['msg'] = val
            ackFut = Future()
            self.messages[ch]['fut'] = ackFut
            self.loopback.send(chIdx)
            ackFut.get()
            self.messages[ch]['msg'] = None
            self.messages[ch]['fut'] = None
            if val == -3 or val == -2:
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
        # print('Done initListen on PE ' + str(charm.myPe()))
        self.level += 1
        self.levelList = []
        self.termCount = 0
        self.levelDone = False
        fut.send(1)
        
    
    @coro
    def listen(self):
        # print('Started main listener')
        msgCount = {}
        for ch in self.inChannels:
            msgCount[ch] = 0
        while any([self.status[ch] > -2 for ch in self.inChannels]):
            chIdx = self.loopback.recv()
            ch = self.inChannels[chIdx]
            msg = self.messages[ch]
            # print('Processed message ' + str(msg))
            val = msg['msg']
            msgCount[ch] += 1
            if val == -3:
                for ch in self.inChannels:
                    if self.status[ch] != -2:
                        self.workerDone[ch].send(1)
                    self.status[ch] = -3
                self.parentChannel.send(-3)
            elif val == -2:
                self.status[ch] = -2
                self.workerDone[ch].send(1)
                self.levelDone = True
            elif type(val) == tuple and len(val) == 3:
                # if self.status[ch] == -1:
                    # Process node
                newNode = self.nodeConstructor(*val, self.localVarGroup[charm.myPe()], self)
                if self.nodeCalls & 1:
                    newNode.init()
                if not newNode in self.table:
                    self.table[newNode] = {'nodeInt': val[2], 'checked':False}
                    self.levelList.append(val[2])
 
            else:
                print('Received unexpected message ' + str(val) + ' on hash worker ' + str(self.thisIndex))
            
            # We're all done with this message, so report back
            localFut = msg['fut']
            self.messages[ch]['fut'] = None
            localFut.send(1)
        # print('Shutting down main listener on PE ' + str(charm.myPe()))
        return 1

    @coro
    def awaitLevel(self):
        for ch in self.inChannels:
            self.workerDone[ch].get()
        self.levelDone = True

    @coro
    def getLevelList(self, levelListFut):
        if self.levelDone:
            self.reduce(levelListFut, self.levelList, Reducer.Join)
        else:
            print('Warning: tried to retrieve level list before level was done!')
            self.reduce(levelListFut, [], Reducer.Join)
    

class DistHash(Chare):

    def __init__(self, feederGroup, nodeConstructor, localVarGroup, hashPEs, posetPEs):
        self.feederGroup = feederGroup
        self.posetPEs = posetPEs
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
        myFut = self.feederGroup.addFeedbackChannel(self.thisProxy, awaitable=True)
        myFut.get()        
        self.feedbackChannels = [Channel(self, remote=proxy) for proxy in feederProxies]
        
        # Establish channels from each feeder worker to each hash worker
        myFut = self.feederGroup.addDestChannel(self.hashWorkerProxies , awaitable=True)
        myFut.get()

        myFut = self.hWorkersFull.addOriginChannel(feederProxies,awaitable=True)
        myFut.get()

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
        self.hWorkers.awaitLevel(awaitable=True).get()
    
    @coro
    def getLevelList(self):
        nextlevel = Future()
        self.hWorkersFull.getLevelList(nextlevel)
        return nextlevel.get()

    def getWorkerProxy(self):
        return self.hWorkers
    @coro
    def initListening(self,allDone):
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