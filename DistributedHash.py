import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
from numpy import may_share_memory
import time
import itertools

def Join(contribs):
    return list(itertools.chain.from_iterable(contribs))
Reducer.addReducer(Join)

class HashWorker(Chare):

    def __init__(self,nodeConstructor,parentProxy):
        self.inChannels = []
        self.level = -1
        self.levelList = []
        self.table = {}
        self.nodeConstructor = nodeConstructor
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
        fut.send(1)
        
    # Protocol codes:
    # -1: initiate a new level of nodes
    # -2: no more new nodes coming on the current level
    # -3: terminate (because the checker found a violating node)
    # Otherwise, we expect a list of the form
    #
    #  (lsbHash, msbHash, nodeInt)
    @coro
    def listen(self):
        # print('Started main listener')
        msgCount = {}
        for ch in self.inChannels:
            msgCount[ch] = 0
        while any([self.status[ch] > -2 for ch in self.inChannels]):
            # print('Main loop')
            chIdx = self.loopback.recv()
            # print(chIdx)
            ch = self.inChannels[chIdx]
            msg = self.messages[ch]
            # print(msg)
            # print(type(msg['fut']))
            # if type(msg['fut']) != charm4py.threads.Future:
            #     print('Continuing')
            #     continue
            # print('Processing message val : ' + str(msg['msg']))
            val = msg['msg']
            msgCount[ch] += 1
            # print('Message number ' + str(msgCount[ch]) + ' on PE ' + str(charm.myPe()) + ' contained ' + str(val) + ' from worker ' + str(senderIdx))
            # senderIdx = 0
            # for ii in range(len(self.inChannels)):
            #     if self.inChannels[ii] == ch:
            #         senderIdx = ii
            #         break
            # print('Message number ' + str(msgCount[ch]) + ' on PE ' + str(charm.myPe()) + ' contained ' + str(val) + ' from worker ' + str(senderIdx))
            # if val == -3:
            #     for ch in self.inChannels:
            #         self.status[ch] = -3
            #     break
            # elif val == -1 and self.status[ch] > -3:
            #     # Set the status for all channels to recieve a new level
            #     if all([self.status[ch] != -1 for ch in self.status]):
                    
            #         for ch in self.status:
            #             self.status[ch] = -1
            if val == -3:
                for ch in self.inChannels:
                    if self.status[ch] != -2:
                        self.workerDone[ch].send(1)
                    self.status[ch] = -3
                # if all([self.status[ch] == -2 for ch in self.status]):
                #     # We have received the final nodes for this level from all feederWorkers
                #     #print('Sending -2 to parent on worker ' + str(charm.myPe()))
                self.parentChannel.send(-3)
            elif val == -2:
                self.status[ch] = -2
                self.workerDone[ch].send(1)
            elif type(val) == tuple and len(val) == 3:
                # if self.status[ch] == -1:
                    # Process node
                newNode = self.nodeConstructor(*val)
                if not newNode in self.table:
                    self.table[newNode] = {'nodeInt': val[2], 'checked':False}
                    self.levelList.append(val[2])
                    # TO DO: Add node checker call here...
                    #print('Received node value ' + str(val))
                # else:
                #     # We're not expecting nodes yet
                #     print('Received unexpected node ' + str(val) + ' on hash worker ' + str(self.thisIndex) + ' [[Non-Listening State:' + str(self.status[ch]) + ']]')
            else:
                print('Received unexpected message ' + str(val) + ' on hash worker ' + str(self.thisIndex))
            
            # We're all done with this message, so report back
            localFut = msg['fut']
            self.messages[ch]['fut'] = None
            localFut.send(1)
        # print('Shutting down main listener on PE ' + str(charm.myPe()))
        return 1

    @coro
    def getLevelList(self, levelListFut):
        for ch in self.inChannels:
            self.workerDone[ch].get()
        self.reduce(levelListFut, self.levelList, Reducer.Join)
    

class DistHash(Chare):

    def __init__(self, feederGroup, nodeConstructor, pelist=None):
        self.feederGroup = feederGroup
        self.nodeConstructor = nodeConstructor
        if pelist == None:
            self.hWorkers = Group(HashWorker,args=[self.nodeConstructor, self.thisProxy])
        else:
            self.hWorkers = Group(HashWorker,args=[self.nodeConstructor, self.thisProxy],onPEs=pelist)
        charm.awaitCreation(self.hWorkers)
        self.hashWorkerProxies = self.hWorkers.getProxies(ret=True)
        self.hashWorkerProxies = self.hashWorkerProxies.get()
        self.hashWorkerChannels = [Channel(self, remote=proxy) for proxy in self.hashWorkerProxies]
        self.hashWorkerStatus = {}
        for ch in self.hashWorkerChannels:
            self.hashWorkerStatus[ch] = 0
        # self.workerIdxs.sort()
        # self.numWorkers = len(self.workerIdxs)
        # for k in range(self.numWorkers):
        #     self.hWorkers[self.workerIdxs[k]].setPos(k)

    @coro
    def initialize(self):
        # Get a list of proxies for all memembers of the feeder group:
        feederProxies = self.feederGroup.getProxies(ret=True)
        feederProxies = feederProxies.get()

        # Establish a feedback channel so that the hash table can send messages to the feeder workers:
        myFut = self.feederGroup.addFeedbackChannel(self.thisProxy, awaitable=True)
        myFut.get()        
        self.feedbackChannels = [Channel(self, remote=proxy) for proxy in feederProxies]
        
        # Establish channels from each feeder worker to each hash worker
        myFut = self.feederGroup.addDestChannel(self.hashWorkerProxies , awaitable=True)
        myFut.get()

        myFut = self.hWorkers.addOriginChannel(feederProxies,awaitable=True)
        myFut.get()

    @coro
    def levelDone(self, doneFut):
        imDone = False
        while not imDone:
            for ch in charm.iwait(self.hashWorkerChannels):
                val = ch.recv()
                if val == -2:
                    self.hashWorkerStatus[ch] = -2
                if all([self.hashWorkerStatus[ch] == -2 for ch in self.hashWorkerChannels]):
                    for ch in self.hashWorkerChannels:
                        self.hashWorkerStatus[ch] = 0
                    doneFut.send(1)
                    imDone = True
                    break
    
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