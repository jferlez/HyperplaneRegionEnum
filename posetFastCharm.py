import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import cdd
import numpy as np
from copy import copy
import time
from itertools import repeat
from functools import partial, total_ordering
import cvxopt
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import warnings

warnings.simplefilter(action = "ignore", category = RuntimeWarning)

class PosetNode:

    def __init__(self, INTrep, level, facesInt=-1):
        self.INTrep = INTrep
        self.level = level
        self.regionProcessed = False
        self.facesInt = facesInt
        self.facesList = []
        self.successors = []


class Poset(Chare):
    
    def __init__(self, checkNodeGroup, groupPEs, useParNodeSched, posetPEs, batchSize):
        
        self.checkNodeGroup = checkNodeGroup
        self.useParNodeSched = useParNodeSched
        self.groupPEs = groupPEs
        self.posetPEs = posetPEs
        self.stackNum = batchSize

        # Create a group to paralellize the computation of successors
        # (Use all PEs unless a list was explicitly passed to us)
        if self.posetPEs == None:
            self.succGroup = Group(successorWorker)
        else:
            self.succGroup = Group(successorWorker, onPEs=self.posetPEs)

        if not self.useParNodeSched:
            self.nodeSchedInst = Chare(checkNodesSchedulerInt, onPE=charm.myPe())
        else:
            self.nodeSchedInst = Chare(checkNodesScheduler, args=[self.thisProxy], onPE=charm.myPe())
            self.nodeCheckChannel = Channel(self, remote=self.nodeSchedInst)

    # @coro
    def initializeFromConstraintObject(self, constraints):
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
        self.populated = False
    
    def initialize(self, AbPairs, pt, fixedA, fixedb):
        self.AbPairs = AbPairs
        self.pt = pt
        self.fixedA = fixedA
        self.fixedb = fixedb

        self.N = len(self.AbPairs[0][0])
        # The number of nodes (regions) in a poset level that are requried to trigger
        # the parallelized code to compute all of the successors:
        self.parallelThreshold = 0

        self.hashTable = {}
        self.levelArray = [[] for i in range(self.N)]

        self.root = PosetNode(intSet(0,self.N),0)
        self.hashTable[self.root.INTrep] = self.root
        self.levelArray[0].append(self.root)
        self.root.regionLeveled = True
        self.incomplete = True
        self.populated = False

    @coro
    def setConstraint(self,lb,out=0):
        self.populated = False
        self.incomplete = True
        self.constraints = constraints( \
                -1*self.AbPairs[out][0], \
                self.AbPairs[out][1] - lb*np.ones((self.N,1)), \
                self.pt, \
                self.fixedA, \
                self.fixedb \
            )
        self.hashTable = {}
        self.levelArray = [[] for i in range(self.N)]

        self.root = PosetNode(intSet(0,self.N),0)
        self.hashTable[self.root.INTrep] = self.root
        self.levelArray[0].append(self.root)
        self.root.regionLeveled = True
        self.incomplete = True
        self.populated = False
        
        return 1


    @coro
    def populatePoset(self, retChannelEndPoint=None, checkNodesFuture=None, method='fastLP', solver='gplk', findAll=False ):
        if self.populated:
            return
        
        
        if method=='cdd':
            processNodeSuccessors = partial(processNodeSuccessorsCDD, solver=solver)
        elif method=='simpleLP':
            processNodeSuccessors = partial(processNodeSuccessorsSimpleLP, solver=solver)
        elif method=='fastLP':
            processNodeSuccessors = partial(processNodeSuccessorsFastLP, solver=solver, findAll=findAll)
        

        emitNodes = False
        if not retChannelEndPoint==None:
            emitNodes = True
            retChannel = Channel(self, remote=retChannelEndPoint)

        checkNodes = False
        if not checkNodesFuture==None:


            if self.checkNodeGroup==None: # or not isinstance(checkNodeGroup,Group):
                raise ValueError('Must supply a Chare Group node-check group via \'checkNodesGroup\' argument!')
            self.peCounter = 0
            self.stackCounter = 0
            self.pes = [i for i in range(charm.numPes())] if len(self.groupPEs)==0 else self.groupPEs
            self.workGroup = [[-1 for i in range(self.stackNum)] for j in range(len(self.pes))]

            checkNodes = True
            returned = False

            # (Re-)Initialize the node checker group with the new data
            
            stat = self.nodeSchedInst.initialize(self.stackNum, self.checkNodeGroup, self.pes, checkNodesFuture,awaitable=True)
            stat.get()
            if  self.useParNodeSched:                
                # This version of nodeSchedInst recieves nodes on a channel, so we need to bring
                # up that channel on the Chare, so it will start processing nodes when we send them on nodeCheckChannel
                nodeReceiverFut = self.nodeSchedInst.receiveNodes(awaitable=True)

        stat = self.succGroup.initialize(self.N,self.constraints.fullConstraints,awaitable=True)
        stat.get()
        self.succGroup.setMethod(method=method,solver=solver,findAll=findAll)

        level = 0
        thisLevel = [0]

        doProcessing = False
        while level < self.N+1 and len(thisLevel) > 0:

            # This is the place to put alternative fast processing of nodes -- e.g. ray shooting to find regions

            # Now parallelize the LPs to find the neihboring regions; we won't put them in the poset
            # yet, though
            if len(thisLevel) < self.parallelThreshold:
                parallelCode = False
            else:
                parallelCode = True

            
            if parallelCode:

                for k in range(charm.numPes()):
                    self.succGroup[k].initList( \
                                [ i for i in thisLevel[k:len(thisLevel):charm.numPes()] ] \
                            )
                transferStatus = Future()
                self.succGroup.collectXferStats(transferStatus)
                stat = transferStatus.get()
                successorList = Future()
                self.succGroup.computeSuccessors(successorList)
                nextLevel = list(successorList.get())

                # Retrieve faces for all the nodes in the current level
                facesList = [0 for i in range(len(thisLevel))]
                for k in range(charm.numPes()):
                    facesListFut = self.succGroup[k].retrieveFaces(awaitable=True)
                    facesListWork = facesListFut.get()
                    for i in range(k,len(thisLevel),charm.numPes()):
                        facesList[i] = facesListWork[int((i-k)/charm.numPes())]

            else:

                successorList = map(processNodeSuccessors, \
                            [node.INTrep.iINT for node in thisLevel], \
                            repeat(self.N), \
                            repeat(self.constraints.fullConstraints) \
                        )
                nextLevel = list(set([]).union(*successorList))



            for k in range(len(thisLevel)):
                i = intSet(thisLevel[k],self.N)
                if i in self.hashTable:
                    thisLevel[k] = self.hashTable[i]
                    thisLevel[k].facesInt = facesList[k]
                else:
                    thisLevel[k] = PosetNode( i, level+1, facesInt=facesList[k] )
                    self.hashTable[i] = thisLevel[k]
                if not thisLevel[k].regionProcessed:
                    if emitNodes:
                        retChannel.send(thisLevel[k].INTrep.iINT)
                    if checkNodes:
                        # First update self.workGroup with the new node
                        if self.peCounter == len(self.pes)-1 and self.stackCounter == self.stackNum - 1:
                            self.workGroup[self.peCounter][self.stackCounter] = thisLevel[k].INTrep.iINT + (thisLevel[k].facesInt << (self.N+1))
                            thisLevel[k].regionProcessed = True
                            self.peCounter += 1
                            doProcessing = True
                        elif self.peCounter < len(self.pes)-1 and self.stackCounter < self.stackNum:
                            self.workGroup[self.peCounter][self.stackCounter] = thisLevel[k].INTrep.iINT + (thisLevel[k].facesInt << (self.N+1))
                            thisLevel[k].regionProcessed = True
                            self.peCounter += 1
                        elif self.peCounter == len(self.pes)-1 and self.stackCounter < self.stackNum - 1:
                            self.workGroup[self.peCounter][self.stackCounter] = thisLevel[k].INTrep.iINT + (thisLevel[k].facesInt << (self.N+1))
                            thisLevel[k].regionProcessed = True
                            self.stackCounter += 1
                            self.peCounter = 0
                        if doProcessing:
                            if not self.useParNodeSched:
                                f = self.nodeSchedInst.checkNode(self.peCounter,self.stackCounter,self.workGroup, awaitable=True)
                                f.get()
                            else:
                                self.nodeCheckChannel.send([self.peCounter,self.stackCounter,copy(self.workGroup)])
                            f = self.nodeSchedInst.foundQ(awaitable=True) 
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
            
            
            
            thisLevel = nextLevel
            level += 1

        if checkNodes:
            if not self.useParNodeSched:
                f = self.nodeSchedInst.checkNode(self.peCounter,self.stackCounter,self.workGroup, awaitable=True)
                finalVal = f.get()
                if not finalVal:
                    checkNodesFuture.send(False)
            else:
                self.nodeCheckChannel.send([self.peCounter,self.stackCounter,copy(self.workGroup)])
                self.nodeCheckChannel.send([])
                nodeReceiverFut.get()

        # Tell the channel endpoint that no more nodes are coming
        if emitNodes:
            retChannel.send(-1)
        self.incomplete = False
        posetLen = 0
        for ii in self.hashTable.keys():
            if self.hashTable[ii].facesInt > 0:
                posetLen += 1
        print('Computed a (partial) poset of size: ' + str(len(self.hashTable.keys())))
        print('Computed a (partial) poset of size (nontrivial regions): ' + str(posetLen))
        # return [i.iINT for i in self.hashTable.keys()]
        self.populated = True
        return posetLen




class successorWorker(Chare):

    def initialize(self,N,fullConstraints):
        self.workInts = []
        self.N = N
        self.fullConstraints = fullConstraints
        self.processNodeSuccessors = partial(processNodeSuccessorsFastLP, solver='gplk')
    
    def setMethod(self,method='fastLP',solver='clp',findAll=True):
        if method=='cdd':
            self.processNodeSuccessors = partial(processNodeSuccessorsCDD, solver=solver)
        elif method=='simpleLP':
            self.processNodeSuccessors = partial(processNodeSuccessorsSimpleLP, solver=solver)
        elif method=='fastLP':
            self.processNodeSuccessors = partial(processNodeSuccessorsFastLP, solver=solver, findAll=findAll)
    
    @coro
    def initList(self,workInts):
        self.status = Future()
        self.workInts = workInts
        self.status.send(1)

    @coro
    def collectXferStats(self, stat_result):
        self.reduce(stat_result, self.status.get() , Reducer.sum)

    @coro
    def computeSuccessors(self, callback):
        successorList = list(map(self.processNodeSuccessors, \
                        self.workInts, \
                        repeat(self.N), \
                        repeat(self.fullConstraints) \
                    )) if len(self.workInts) > 0 else [[set([]),-1]]
        self.workInts = [successorList[ii][1] for ii in range(len(successorList))]
        successorList = [successorList[ii][0] for ii in range(len(successorList))]
        self.reduce(callback, set([]).union(*successorList), Reducer.Union)

    @coro
    def retrieveFaces(self):
        return self.workInts




class checkNodesSchedulerInt(Chare):
    
    def initialize(self, stackNum, checkNodeGroup, groupPEs, retFuture):
        self.stackNum = stackNum
        self.checkNodeChareGroup = checkNodeGroup
        self.retFuture = retFuture
        self.pes = groupPEs
        # self.waitOn = waitOn
        
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

    def __init__(self, channelEndpoint):
        self.nodeChannel = Channel(self,remote=channelEndpoint)
    
    def initialize(self, stackNum, checkNodeGroup, groupPEs, retFuture):
        
        self.stackNum = stackNum
        self.checkNodeChareGroup = checkNodeGroup
        self.retFuture = retFuture
        self.pes = groupPEs
        
        self.wrkGrpFuture = None
        self.found = False

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





        



def Union(contribs):
    return set().union(*contribs)

Reducer.addReducer(Union)


def processNodeSuccessorsCDD(INTrep,N,H2,solver='gplk'):
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
        to_keep = concreteMinHRep(H,copyMat=False,solver=solver)
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
    
    return [set(successors), facesInt]


def processNodeSuccessorsSimpleLP(INTrep,N,H2,solver='gplk'):
    H = copy(H2)
    # global H2
    # H = np.array(H2)
    # H = np.array(processNodeSuccessors.H)
    idx = 1
    for i in range(N):
        if INTrep & idx > 0:
            H[i] = -1*H[i]
        idx = idx << 1
    
    to_keep = concreteMinHRep(H,copyMat=False,solver=solver)
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
    
    return [set(successors), facesInt]


def concreteMinHRep(H2,cnt=None,randomize=False,copyMat=True,solver='gplk'):
    if not randomize:
        if copyMat:
            H = copy(H2)
        else:
            H = H2
    else:
        H = H2[np.random.permutation(len(H2)),:]
    if cnt is None:
        cntr = len(H)
    else:
        cntr = cnt

    d = H.shape[1]-1
    
    if solver=='clp':
        s = CyClpSimplex()
        xVar = s.addVariable('x', d)
        s.logLevel = 0

    idx = 0
    loc = 0
    e = np.zeros((len(H),1))
    to_keep = list(range(len(H)))
    while idx < len(H) and cntr > 0:
        if solver=='gplk':
            e[idx,0] = 1
            cvxArgs = [cvxopt.matrix(H[idx,1:]), \
                    cvxopt.matrix(-np.vstack([H[to_keep,1:], [-H[idx,1:]]])), \
                    cvxopt.matrix(np.hstack([H[to_keep,0]+e[to_keep,0], [-H[idx,0]]])) \
                    ]
            e[idx,0] = 0
            sol = cvxopt.solvers.lp(*cvxArgs,solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
            status = sol['status']
            x = sol['x']
        elif solver=='clp':
            e[idx,0] = 1
            for constr in range(len(s.constraints)):
                s.removeConstraint(s.constraints[constr].name)
            s += np.matrix(-np.vstack([H[to_keep,1:], [-H[idx,1:]]])) * xVar <= \
                CyLPArray((np.hstack([H[to_keep,0]+e[to_keep,0], [-H[idx,0]]])).flatten())
            s.objective = CyLPArray(H[idx,1:])
            e[idx,0] = 0
            try:
                status = s.primal()
                x = np.array(s.primalVariableSolution['x']).reshape((d,1))
            except:
                # Something went wrong with the CLP solver, so force use of GLPK
                print(' ')
                print('********************  PE' + str(charm.myPe()) + ' WARNING!!  ********************')
                print('PE' + str(charm.myPe()) + ': Needed to fallback to GLPK for unknown reasons!!' ) 
                print(' ')
                status = 'unk'
        if status != 'optimal' and status != 'primal infeasible':
            # If we chose Clp as a solver, use GPLK as a fallback
            if solver=='clp':
                e[idx,0] = 1
                #cvxArgs = [cvxopt.matrix(H[idx,1:]), cvxopt.matrix(-H[to_keep,1:]), cvxopt.matrix(H[to_keep,0]+e[to_keep,0])]
                cvxArgs = [cvxopt.matrix(H[idx,1:]), \
                    cvxopt.matrix(-np.vstack([H[to_keep,1:], [-H[idx,1:]]])), \
                    cvxopt.matrix(np.hstack([H[to_keep,0]+e[to_keep,0], [-H[idx,0]]])) \
                    ]
                e[idx,0] = 0
                sol = cvxopt.solvers.lp(*cvxArgs,solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
                status = sol['status']
                x = sol['x']
            if status != 'optimal' and status != 'primal infeasible':
                print('********************  PE' + str(charm.myPe()) + ' WARNING!!  ********************')
                print('PE' + str(charm.myPe()) + ': Infeasible or numerical ill-conditioning detected at node' )
                print('PE ' + str(charm.myPe()) + ': RESULTS MAY NOT BE ACCURATE!!')
                return [set([]), 0]
        if status == 'primal infeasible' or np.all(-H[to_keep,1:]@x <= H[to_keep,0].reshape((len(to_keep),1))):
            # inequality is redundant, so remove it
            to_keep.pop(loc)
            cntr -= 1
        else:
            loc += 1
            cntr -= 1
        idx += 1
    return to_keep[0:min(loc if not cnt is None else len(to_keep),len(to_keep))]


def processNodeSuccessorsFastLP(INTrep,N,H2,solver='gplk',findAll=False):

    # H = copy(H2)
    # global H2
    # H = np.array(H2)
    # H = np.array(processNodeSuccessors.H)
    flippable = np.zeros((N,),dtype=np.int32)
    unflippable = np.zeros((N,),dtype=np.int32)
    flipIdx = 0
    unflipIdx = 0
    idx = 1
    for i in range(N):
        if INTrep & idx > 0:
            # H[i] = -1*H[i]
            unflippable[unflipIdx] = i
            unflipIdx += 1
        else:
            flippable[flipIdx] = i
            flipIdx += 1
        idx = idx << 1
    # flippable = flippable[0:flipIdx]
    # unflippable = unflippable[0:unflipIdx]
    

    if not findAll:
        # Now all of the flippable hyperplanes will be at the beginning
        # flippable = sorted(list(set(range(N))-set(unflippable)))
        H = H2[np.hstack([flippable[0:flipIdx], unflippable[0:unflipIdx]]),:]
        reorder = np.hstack([flippable[0:flipIdx], unflippable[0:unflipIdx], np.array(range(N,H2.shape[0]),dtype=np.int32)])
        H[flipIdx:,:] = -H[flipIdx:,:]
        H3 = np.vstack([H, H2[N:,:]])
        H=H3
    else:
        H = copy(H2)
        H[unflippable[0:unflipIdx],:] = -H[unflippable[0:unflipIdx],:]
    
    d = H.shape[1]-1

    if solver=='clp':
        s = CyClpSimplex()
        xVar = s.addVariable('x', d)
        s.logLevel = 0
    
    doBounding = False
    # Don't compute the bounding box if the number of flippable hyperplanes is almost 2*d,
    # since we have to do 2*d LPs just to get the bounding box
    if not findAll and len(flippable) > 3*d:
        doBounding = True
    # If we want all the faces, we should decide whether to compute the bounding box based on
    # the number N instead:
    if findAll and N > 3*d:
        doBounding = True

    if doBounding:
        # Find a bounding box
        bbox = [[] for ii in range(d)]
        ed = np.zeros((d,1))
        for ii in range(d):
            for direc in [1,-1]:
                if solver=='gplk':
                    ed[ii,0] = direc
                    cvxArgs = [cvxopt.matrix(ed), cvxopt.matrix(-H[:,1:]), cvxopt.matrix(H[:,0])]
                    ed[ii,0] = 0
                    sol = cvxopt.solvers.lp(*cvxArgs,solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
                    status = sol['status']
                    x = sol['x']
                elif solver=='clp':
                    ed[ii,0] = direc
                    for constr in range(len(s.constraints)):
                        s.removeConstraint(s.constraints[constr].name)
                    s +=  np.matrix(-H[:,1:]) * xVar <= CyLPArray(H[:,0])
                    s.objective = CyLPArray(ed.flatten())
                    ed[ii,0] = 0
                    status = s.primal()
                    x = np.array(s.primalVariableSolution['x']).reshape((d,1))
                # In case we have problems with Clp, use GPLK as as fallback
                if solver =='clp' and status != 'optimal' and status != 'dual infeasible':
                    ed[ii,0] = direc
                    cvxArgs = [cvxopt.matrix(ed), cvxopt.matrix(-H[:,1:]), cvxopt.matrix(H[:,0])]
                    ed[ii,0] = 0
                    sol = cvxopt.solvers.lp(*cvxArgs,solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
                    status = sol['status']
                    x = sol['x']
                if status == 'optimal':
                    bbox[ii].append(np.array(x[ii,0]))
                elif status == 'dual infeasible':
                    bbox[ii].append(-1*direc*np.inf)
                else:
                    print('********************  PE' + str(charm.myPe()) + ' WARNING!!  ********************')
                    print('PE' + str(charm.myPe()) + ': Infeasible or numerical ill-conditioning detected while computing bounding box!')
                    return [set([]), 0]

        boxCorners = np.array(np.meshgrid(*bbox)).T.reshape(-1,d).T

        to_keep = np.nonzero(np.any(((-H[:,1:] @ boxCorners) - H[:,0].reshape((len(H),1))) >= -1e-07,axis=1))[0]
    else:
        to_keep = np.array(range(H.shape[0]),dtype=np.int32)
    
    if not findAll:
        findSize = 0
        for ii in range(len(to_keep)):
            if to_keep[ii] >= flipIdx:
                break
            findSize += 1
    else:
        findSize = None
    
    # to_keep = to_keep.tolist()

    idx = 0
    loc = 0
    e = np.zeros((len(H),1))
    
    to_keep_sub = concreteMinHRep(H[to_keep,:],cnt=findSize,copyMat=False,solver=solver)
    if findSize is None:
        findSize = len(to_keep)
    to_keep_faces = to_keep[to_keep_sub].tolist() + to_keep[findSize:len(to_keep)].tolist()
    to_keep = to_keep[to_keep_sub].tolist()

    if not findAll:
        to_keep = reorder[to_keep].tolist()
        to_keep_faces = reorder[to_keep_faces].tolist()

    # print([findSize, len(to_keep), len(to_keep_faces)])

    # Use this to keep track of the region's faces
    facesInt = 0
    for k in to_keep_faces:
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
    
    return [set(successors), facesInt]



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

def unflipInt(INT, flipSet,N):
    retInt = INT
    if N >= 63:
        shift = 2**(2*N+2)
    else:
        shift = 0
    for k in flipSet:
        # Force sel to be considered a long integer
        sel = shift + (1 << k)
        if INT & sel > 0:
            retInt -= sel
        else:
            retInt += sel
    return retInt & (2**(N+2) - 1)

def unflipIntFixed(INT, flipSet,N):
    retInt = 0
    mask = (2**(N+1) - 1)
    if N >= 63:
        shift = 2**(2*N+2)
    else:
        shift = 0
    for k in range(N):
        sel = shift + (1 << k)
        if k in flipSet:
            if (sel & INT & mask) > 0:
                retInt += sel
        else:
            if (sel & INT & mask) == 0:
                retInt += sel
    return retInt & mask

# This function is TOTALLY misnamed: it actually returns the set of NEGATIVE hyperplanes....
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


def activeHyperplaneSet(INT,n):
    retList = [-1 for i in range(n)]
    retIdx = 0
    idx = 1
    for i in range(n):
        if INT & idx > 0:
            retList[retIdx] = i
            retIdx += 1
        idx = idx << 1
    return retList[0:retIdx]