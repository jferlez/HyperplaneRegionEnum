import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import cdd
import numpy as np
from copy import copy
import time
from itertools import repeat



class PosetNode:

    def __init__(self, INTrep, level):
        self.INTrep = INTrep
        self.level = level
        self.regionProcessed = False
        self.facesInt = 0
        self.facesList = []
        self.successors = []


class Poset(Chare):
    @coro
    def __init__(self, constraints):
        self.constraints = constraints
        self.N = len(self.constraints.nA)
        # The number of nodes (regions) in a poset level that are requried to trigger
        # the parallelized code to compute all of the successors:
        self.parallelThreshold = 0

        self.hashTable = {}
        self.levelArray = [[] for i in range(len(self.constraints.nA))]

        self.root = PosetNode(intSet(0,self.N), self.constraints)
        self.hashTable[self.root.INTrep] = self.root
        self.levelArray[0].append(self.root)
        self.root.regionLeveled = True
        self.incomplete = True
        self.stackNum = 10
        
    
    @coro
    def populatePoset(self, retChannelEndPoint=None, checkNodesFuture=None, checkNodeGroup=None, groupPEs=[], useParNodeSched=False, posetPEs=None):

        emitNodes = False
        if not retChannelEndPoint==None:
            emitNodes = True
            retChannel = Channel(self, remote=retChannelEndPoint)

        checkNodes = False
        if not checkNodesFuture==None:


            if checkNodeGroup==None: # or not isinstance(checkNodeGroup,Group):
                raise ValueError('Must supply a Chare Group node-check group via \'checkNodesGroup\' argument!')
            self.peCounter = 0
            self.stackCounter = 0
            self.pes = [i for i in range(charm.numPes())] if len(groupPEs)==0 else groupPEs
            self.workGroup = [[-1 for i in range(self.stackNum)] for j in range(len(self.pes))]

            checkNodes = True
            returned = False

            if not useParNodeSched:
                nodeSchedInst = Chare(checkNodesSchedulerInt, args=[self.stackNum, checkNodeGroup, self.pes, checkNodesFuture, True], onPE=charm.myPe())
            else:
                nodeSchedInst = Chare(checkNodesScheduler, args=[self.stackNum, checkNodeGroup, self.pes, checkNodesFuture, self.thisProxy], onPE=charm.myPe())
                nodeCheckChannel = Channel(self, remote=nodeSchedInst)
                # This version of nodeSchedInst recieves nodes on a channel, so we need to bring
                # up that channel on the Chare, so it will start processing nodes when we send them on nodeCheckChannel
                nodeReceiverFut = nodeSchedInst.receiveNodes(awaitable=True)

        # Create a group to paralellize the computation of successors
        # (Use all PEs unless a list was explicitly passed to us)
        if posetPEs == None:
            succGroup = Group(successorWorker,args=[self.N,self.constraints.fullConstraints])
        else:
            succGroup = Group(successorWorker,args=[self.N,self.constraints.fullConstraints], onPEs=posetPEs)

        level = 0
        thisLevel = [0]

        doProcessing = False
        while level < self.N and len(thisLevel) > 0:

            # This is the place to put alternative fast processing of nodes -- e.g. ray shooting to find regions

            for k in range(len(thisLevel)):
                i = intSet(thisLevel[k],self.N)
                if i in self.hashTable:
                    thisLevel[k] = self.hashTable[i]
                else:
                    thisLevel[k] = PosetNode( i, level+1 )
                    self.hashTable[i] = thisLevel[k]
                if not thisLevel[k].regionProcessed:
                    if emitNodes:
                        retChannel.send(thisLevel[k].INTrep.iINT)
                    if checkNodes:
                        # First update self.workGroup with the new node
                        if self.peCounter == len(self.pes)-1 and self.stackCounter == self.stackNum - 1:
                            self.workGroup[self.peCounter][self.stackCounter] = thisLevel[k].INTrep.iINT
                            doProcessing = True
                        elif self.peCounter < len(self.pes)-1 and self.stackCounter < self.stackNum:
                            self.workGroup[self.peCounter][self.stackCounter] = thisLevel[k].INTrep.iINT
                            self.peCounter += 1
                        elif self.peCounter == len(self.pes)-1 and self.stackCounter < self.stackNum - 1:
                            self.workGroup[self.peCounter][self.stackCounter] = thisLevel[k].INTrep.iINT
                            self.stackCounter += 1
                            self.peCounter = 0
                        if doProcessing:
                            if not useParNodeSched:
                                f = nodeSchedInst.checkNode(self.peCounter,self.stackCounter,self.workGroup, awaitable=True)
                                f.get()
                            else:
                                nodeCheckChannel.send([self.peCounter,self.stackCounter,copy(self.workGroup)])
                            f = nodeSchedInst.foundQ(awaitable=True) 
                            if f.get():
                                self.incomplete = True
                                # We found a 'True' on some poset node, so shut everything down
                                if emitNodes:
                                    retChannel.send(-2)
                                return
                            # Reset the counters
                            doProcessing = False
                            self.peCounter = 0
                            self.stackCounter = 0

            # Now parallelize the LPs to find the neihboring regions; we won't put them in the poset
            # yet, though
            if len(thisLevel) < self.parallelThreshold:
                parallelCode = False
            else:
                parallelCode = True

            
            if parallelCode:
                for k in range(charm.numPes()):
                    # print([ i.INTrep.iINT for i in thisLevel[k:len(thisLevel):charm.numPes()] ])
                    succGroup[k].initList( \
                                [ i.INTrep.iINT for i in thisLevel[k:len(thisLevel):charm.numPes()] ] \
                            )
                transferStatus = Future()
                succGroup.collectXferStats(transferStatus)
                cnt = transferStatus.get()

                successorList = Future()
                succGroup.computeSuccessors(successorList)
                nextLevel = list(successorList.get())
            else:
                successorList = map(processNodeSuccessors, \
                            [node.INTrep.iINT for node in thisLevel], \
                            repeat(self.N), \
                            repeat(self.constraints.fullConstraints) \
                        )
                nextLevel = list(set([]).union(*successorList))
            
            

            
            thisLevel = nextLevel
            level += 1

        if checkNodes:
            if not useParNodeSched:
                f = nodeSchedInst.checkNode(self.peCounter,self.stackCounter,self.workGroup, awaitable=True)
                finalVal = f.get()
                if not finalVal:
                    checkNodesFuture.send(False)
            else:
                nodeCheckChannel.send([self.peCounter,self.stackCounter,copy(self.workGroup)])
                nodeCheckChannel.send([])
                nodeReceiverFut.get()
        # print('made it to here')
        # Tell the channel endpoint that no more nodes are coming
        if emitNodes:
            retChannel.send(-1)
        self.incomplete = False
        print('Computed a (partial) poset of size: ' + str(len(self.hashTable.keys())))
        # return [i.iINT for i in self.hashTable.keys()]
        return 0




class successorWorker(Chare):

    def __init__(self,N,fullConstraints):
        self.workInts = []
        self.N = N
        self.fullConstraints = fullConstraints
    
    @coro
    def initList(self,workInts):
        self.status = Future()
        self.workInts = workInts
        self.status.send(1)

    @coro
    def collectXferStats(self, stat_result):
        self.reduce(stat_result, self.status.get(), Reducer.sum)

    @coro
    def computeSuccessors(self, callback):
        successorList = map(processNodeSuccessors, \
                        self.workInts, \
                        repeat(self.N), \
                        repeat(self.fullConstraints) \
                    ) if len(self.workInts) > 0 else [set([])]
        self.workInts = []

        self.reduce(callback, set([]).union(*successorList), Reducer.Union)




class checkNodesSchedulerInt(Chare):
    
    def __init__(self, stackNum, checkNodeGroup, groupPEs, retFuture, waitOn):
        self.stackNum = stackNum
        self.checkNodeChareGroup = checkNodeGroup
        self.retFuture = retFuture
        self.pes = groupPEs
        self.waitOn = waitOn
        
        self.wrkGrpFuture = None
        self.found = False

    @coro
    def foundQ(self):
        return self.found
    
    @coro
    def checkNode(self, peCounter, stackCounter, workGroup):
        if self.found:
            return

        for peIdx in range(len(self.pes)):
            # print('Transferring worklist to PE ' + str(peIdx))
            # print(workGroup[peIdx][ 0:(stackCounter+1 if peIdx < peCounter else stackCounter) ])
            self.checkNodeChareGroup[self.pes[peIdx]].initList( \
                    workGroup[peIdx][ 0:(stackCounter+1 if peIdx < peCounter else stackCounter) ]
                )
            
        xferFut = Future()
        self.checkNodeChareGroup.collectXferStats(xferFut)
        xferFut.get()

        localFuture = Future()
        self.checkNodeChareGroup.check(localFuture, awaitable=True)
        retVal = localFuture.get()
        if retVal:
            self.retFuture.send(True)
            self.found = True
        return retVal
        

# Shares several methods wtih checkNodesSchedulerInt, but subclassing Chares doesn't seem to work
class checkNodesScheduler(Chare):

    def __init__(self, stackNum, checkNodeGroup, groupPEs, retFuture, channelEndpoint):
        # super().__init__(stackNum, checkNodeGroup, groupPEs, retFuture, False)
        self.stackNum = stackNum
        self.checkNodeChareGroup = checkNodeGroup
        self.retFuture = retFuture
        self.pes = groupPEs
        # Specific to this class:
        self.waitOn = False
        
        self.wrkGrpFuture = None
        self.found = False
        self.nodeChannel = Channel(self,remote=channelEndpoint)

    # Should be the same method as checkNodesSchedulerInt
    @coro
    def foundQ(self):
        return self.found
    
    # Should be the same method as checkNodesSchedulerInt
    @coro
    def checkNode(self, peCounter, stackCounter, workGroup):
        if self.found:
            return

        for peIdx in range(len(self.pes)):
            # print('Transferring worklist to PE ' + str(peIdx))
            # print(workGroup[peIdx][ 0:(stackCounter+1 if peIdx < peCounter else stackCounter) ])
            self.checkNodeChareGroup[self.pes[peIdx]].initList( \
                    workGroup[peIdx][ 0:(stackCounter+1 if peIdx < peCounter else stackCounter) ] \
                )
            
        xferFut = Future()
        self.checkNodeChareGroup.collectXferStats(xferFut)
        xferFut.get()

        localFuture = Future()
        self.checkNodeChareGroup.check(localFuture, awaitable=True)
        retVal = localFuture.get()
        if retVal:
            self.retFuture.send(True)
            self.found = True
        return retVal

    @coro
    def receiveNodes(self):
        f = Future()
        f.send(1)
        while True:
            val = self.nodeChannel.recv()
            if len(val) == 0:
                # Wait for the last workgroup to finish
                f.get()
                if not self.found:
                    self.retFuture.send(False)
                break
            else:
                f.get()
                if self.found:
                    break
                f = self.thisProxy.checkNode(*val,awaitable=True)






class CheckNodes(Chare):
    def __init__(self,N,fullConstraints):
        self.N = N
        self.fullConstraints = fullConstraints
        self.workInts = None
    
    @coro
    def initList(self,workInts):
        self.workInts = workInts

    @coro
    def doCheckNode(self,fn,args,chunkFuture):
        found = self.found
        if found:
            return False
        val = fn(self.workInts,self.fullConstraints,*args)
        if val and not self.found:
            self.found = True
            self.foundFuture.sent(True)
        self.reduce(chunkFuture, val, Reducer.logical_or)


        



def Union(contribs):
    return set().union(*contribs)

Reducer.addReducer(Union)


def processNodeSuccessors(INTrep,N,H2):
    H = copy(H2)
    # global H2
    # H = np.array(H2)
    # H = np.array(processNodeSuccessors.H)
    idx = 1
    for i in range(N):
        if INTrep & idx > 0:
            H[i] = -1*H[i]
        idx = idx << 1
    
    mat = cdd.Matrix(H)
    mat.rep_type = cdd.RepType.INEQUALITY
    ret = mat.canonicalize()
    to_keep = sorted(list(frozenset(range(len(H))) - ret[1]))
    # successors[idxOut].clear()
    successors = []
    for i in range(len(to_keep)):
        if to_keep[i] >= N:
            break
        idx = 1 << to_keep[i]
        if idx & INTrep <= 0:
            successors.append( \
                    INTrep + idx \
                )
    
    # Use this if we need to keep track of the region's faces
    # facesInt = 0
    # for k in to_keep:
    #     facesInt = facesInt + (1 << k)
    # successors.append(facesInt)
    return set(successors)



def prepareGlobal(pe,constraints,myId):
    pe.updateGlobals({'FULLCONSTRAINTS_'+str(myId): constraints}, module_name='posetFastCharm', awaitable=True)





class constraints:

    def __init__(self, nA, nb, pt, fA=None, fb=None):
        v = nA @ pt
        v = v.flatten() - nb.flatten()
        self.flipMapN = np.where(v<0,-1,1)
        self.flipMapSet = frozenset(np.nonzero(self.flipMapN < 0)[0])
        self.nA = np.diag(self.flipMapN) @ nA
        self.nb = np.diag(self.flipMapN) @ nb
        self.N = len(nA)

        if (fA is not None) and (fb is not None):
            v = fA @ pt
            v = v.flatten() - fb.flatten()
            if len(np.flatnonzero(v<0)) > 0:
                raise ValueError('Supplied point must satisfy all specified \'fixed\' constraints -- i.e. fA @ pt >= fb !')
            self.fA = fA
            self.fb = fb
            self.fullConstraints = np.vstack( ( np.hstack((-1*self.nb,self.nA)), np.hstack((-1*self.fb,self.fA)) ) )

        else:
            self.fA = None
            self.fb = None
            self.fullConstraints = np.hstack((-self.nb,self.nA))




class intSet:
    def __init__(self, i, n):
        self.iINT = i
        self.n = n
    def __hash__(self):
        p = 6148914691236517205*(self.iINT^(self.iINT>>32))
        return 17316035218449499591*(p^(p>>32))
        # p = self.iINT
        # return p
    def __eq__(self,other):
        if type(other) == int:
            return self.iINT == other
        else:
            return self.iINT == other.iINT
    def getList(self,flipMap):
        retList = [1 for i in range(self.n)]
        idx = 1
        for i in range(self.n):
            if self.iINT & idx > 0:
                retList[i] = -1*flipMap[i]
            else:
                retList[i] = flipMap[i]
            idx = idx << 1
        return retList
    def getInt(self,flipMap):
        retInt = self.iINT
        idx = 1
        for i in range(self.n):
            if flipMap[i] < 0:
                retInt = retInt^idx
            idx = idx << 1
        return retInt
    def flipElement(self,k):
        print('2')


def makeList(INT,n):
    retList = [1 for i in range(n)]
    idx = 1
    for i in range(n):
        if INT & idx > 0:
            retList[i] = -1
        idx = idx << 1
    return retList

def convertToList(INT,n):
    retList = [1 for i in range(n)]
    idx = 1
    for i in range(n):
        if INT & idx > 0:
            retList[i] = -1
        idx = idx << 1
    return retList

def unflipInt(INT, flipSet):
    retInt = INT
    for k in flipSet:
        sel = 1 << k
        if INT & sel > 0:
            retInt -= sel
        else:
            retInt += sel
    return retInt

def posHyperplaneSet(INT,n):
    retList = [-1 for i in range(n)]
    retIdx = 0
    idx = 1
    for i in range(n):
        if INT & idx == 0:
            retList[retIdx] = i
            retIdx += 1
        idx = idx << 1
    return retList[0:retIdx]