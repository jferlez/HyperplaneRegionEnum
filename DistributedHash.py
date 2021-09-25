import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
from numpy import may_share_memory
import time

class PosetNode():

    def __init__(self,lsb,msb,nodeInt,facesInt):
        self.lsbHash = lsb
        self.msbHash = msb
        self.nodeInt = nodeInt
        self.facesInt = facesInt
    
    def __hash__(self):
        return self.msbHash


class HashWorker(Chare):

    def __init__(self):
        self.inChannels = []
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

    @coro
    def listen(self):
        while True:
            for ch in charm.iwait(self.inChannels):
                val = ch.recv()
                print('Received ' + str(val) + ' on hash worker ' + str(self.thisIndex))
        

class DistHash(Chare):

    def __init__(self, feederGroup, pelist=None):
        self.feederGroup = feederGroup
        if pelist == None:
            self.hWorkers = Group(HashWorker)
        else:
            self.hWorkers = Group(HashWorker,onPEs=pelist)
        charm.awaitCreation(self.hWorkers)
        self.workerProxies = self.hWorkers.getProxies(ret=True)
        self.workerProxies = self.workerProxies.get()
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
        myFut = self.feederGroup.addDestChannel(self.workerProxies, awaitable=True)
        myFut.get()

        myFut = self.hWorkers.addOriginChannel(feederProxies,awaitable=True)
        myFut.get()

    
    def getWorkerProxy(self):
        return self.hWorkers
