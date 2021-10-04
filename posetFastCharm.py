import charm4py
from charm4py import charm, Chare, coro, Reducer, Group, Future, Array, Channel
import cdd
import numpy as np
from copy import copy
import time
import itertools
from functools import partial
import encapsulateLP
import DistributedHash
import cvxopt
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import hashlib
import sys
import warnings

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
        self.constraints = self.localProxy.getConstraints(ret=True).get()

    # def update(self):
    #     pass

    # def check(self):
    #     pass

class localVar(Chare):
    def __init__(self,constraints):
        self.constraints = constraints
    def getConstraints(self):
        return self.constraints

class Poset(Chare):
    
    def __init__(self, peSpec, batchSize, nodeConstructor, localVarGroup):
        
        self.stackNum = batchSize
        # To do: check to make sure we're passed a valid Group in localVarGroup
        self.localVarGroup = localVarGroup
        self.useDefaultLocalVarGroup = False
        if localVarGroup is None:
            self.useDefaultLocalVarGroup = True
        self.nodeConstructor = nodeConstructor
        if self.nodeConstructor is None:
            self.nodeConstructor = PosetNode
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
        self.succGroupFull = Group(successorWorker,args=[self.posetPElist])
        secs = [self.succGroupFull[r[0]:r[1]:r[2]] for r in self.posetPEs]
        self.succGroup = charm.combine(*secs)
        successorProxies = self.succGroupFull.getProxies(ret=True).get()
        self.successorProxies = list(itertools.chain.from_iterable( \
                [successorProxies[r[0]:r[1]:r[2]] for r in self.posetPEs] \
            ))

    
    def initialize(self, AbPairs, pt, fixedA, fixedb):
        self.AbPairs = AbPairs
        self.pt = pt
        self.fixedA = fixedA
        self.fixedb = fixedb

        self.N = len(self.AbPairs[0][0])


    @coro
    def setConstraint(self,lb=0,out=0):
        self.populated = False
        self.incomplete = True
        self.flippedConstraints = flipConstraintsReduced( \
                -1*self.AbPairs[out][0], \
                self.AbPairs[out][1] - lb*np.ones((self.N,1)), \
                self.pt, \
                self.fixedA, \
                self.fixedb \
            )
        
        stat = self.succGroup.initialize(self.N,self.flippedConstraints.constraints,awaitable=True)
        stat.get()
        if self.useDefaultLocalVarGroup:
            self.localVarGroup = Group(localVar,args=[self.flippedConstraints])
            charm.awaitCreation(self.localVarGroup) 
        # Initialize a new distributed hash table:
        self.distHashTable = Chare(DistributedHash.DistHash,args=[
            self.succGroupFull, \
            self.nodeConstructor, \
            self.localVarGroup, \
            self.hashPEs, \
            self.posetPEs \
        ])
        # print('Initialized distHashTable group')
        initFut = self.distHashTable.initialize(awaitable=True)
        initFut.get()

        self.populated = False

        return 1


    @coro
    def populatePoset(self, method='fastLP', solver='clp', findAll=False ):
        if self.populated:
            return
        

        self.succGroup.setMethod(method=method,solver=solver,findAll=findAll)

        
        #self.succGroup.testSend()


        level = 0
        thisLevel = [self.flippedConstraints.root]
        posetLen = 1

        # Send this node into the distributed hash table and check it
        initFut = Future()
        self.distHashTable.initListening(initFut)
        initFut.get()


        self.succGroup.resetMessageState(awaitable=True).get()

        self.successorProxies[0].hashAndSend(thisLevel[0],ret=True).get()

        self.succGroup.sendAll(-2)
        self.succGroup.flushMessages(ret=True).get()
        
        # print('Done sending message on RateChannel')
        self.distHashTable.levelDone(awaitable=True).get()
        # print('Waiting for level done')
        while level < self.N+1 and len(thisLevel) > 0:

            # successorProxies = self.succGroup.getProxies(ret=True).get()
            doneFuts = [Future() for k in range(len(self.successorProxies))]
            for k in range(len(self.successorProxies)):
                self.successorProxies[k].initListNew( \
                            [ i for i in thisLevel[k:len(thisLevel):len(self.posetPElist)] ], \
                            doneFuts[k]
                        )
            cnt = 0
            for fut in charm.iwait(doneFuts):
                cnt += fut.get()

            initFut = Future()
            self.distHashTable.initListening(initFut)
            initFut.get()

            self.succGroup.resetMessageState(awaitable=True).get()

            self.succGroup.computeSuccessorsNew(awaitable=True).get()

            self.succGroup.flushMessages(ret=True).get()

            # print('Started looking for successors')
            checkVal = self.distHashTable.levelDone(ret=True).get()
            if not checkVal:
                break
            # print('Done with level')
            nextLevel = self.distHashTable.getLevelList(ret=True).get()

            

            posetLen += len(nextLevel)
            # print(posetLen)
            # Retrieve faces for all the nodes in the current level
            facesList = [0 for i in range(len(thisLevel))]
            for k in range(len(self.posetPElist)):
                facesListFut = self.succGroupFull[self.posetPElist[k]].retrieveFaces(awaitable=True)
                facesListWork = facesListFut.get()
                for i in range(k,len(thisLevel),len(self.posetPElist)):
                    facesList[i] = facesListWork[int((i-k)/len(self.posetPElist))]


            
            thisLevel = nextLevel
            level += 1

        # Note, this print has to go here because this coroutine is only suspending until checkNodes is set
        lpCountFut = Future()
        self.succGroupFull.getLPCount(lpCountFut)
        lpCount = lpCountFut.get()
        print('Total LPs used: ' + str(lpCount))

        print('Checker returned value: ' + str(checkVal))
        
        # print('Computed a (partial) poset of size: ' + str(len(self.hashTable.keys())))
        print('Computed a (partial) poset of size: ' + str(posetLen))
        # return [i.iINT for i in self.hashTable.keys()]
        self.populated = True
        return posetLen




class successorWorker(Chare):

    def __init__(self,pes):
        self.posetPElist = pes

    def initialize(self,N,constraints):
        self.workInts = []
        self.N = N
        self.constraints = constraints
        self.processNodeSuccessors = partial(successorWorker.processNodeSuccessorsFastLP, self, solver='glpk')
        # Defaults to glpk, so this empty call is ok:
        self.lp = encapsulateLP.encapsulateLP()
        self.outChannels = []
        self.endian = sys.byteorder
    
    def setMethod(self,method='fastLP',solver='clp',findAll=True):
        self.lp.initSolver(solver=solver, opts={'dim':len(self.constraints[0])-1})
        if method=='cdd':
            self.processNodeSuccessors = partial(successorWorker.processNodeSuccessorsCDD, self, solver=solver)
        elif method=='fastLP':
            self.processNodeSuccessors = partial(successorWorker.processNodeSuccessorsFastLP, self, solver=solver, findAll=findAll)

        self.solver = solver
        self.findAll = findAll

    @coro
    def getLPCount(self, lpCountFut):
        retVal = 0 if not charm.myPe() in self.posetPElist else self.lp.lpCount
        self.reduce(lpCountFut,retVal,Reducer.sum)
    
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
        if self.N % 4 == 0:
            self.numBytes = self.N/4
        else:
            self.numBytes = int(self.N/4)+1
        # print(self.outChannels)
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

    def hashNodeMD5(self,nodeInt):
        hashInt = int.from_bytes( \
            hashlib.md5(nodeInt.to_bytes(self.numBytes,byteorder=self.endian)).digest(), \
            byteorder=self.endian \
        )
        return ( (hashInt & self.hashMask) % self.numHashWorkers , hashInt >> self.numHashBits, nodeInt )
    
    def hashNode(self,nodeInt):
        p = 6148914691236517205*(nodeInt^(nodeInt>>32))
        hashInt = (17316035218449499591*(p^(p>>32))) & ((1 << 33)-1)
        return ( (hashInt & self.hashMask) % self.numHashWorkers , hashInt >> self.numHashBits, nodeInt )
    @coro
    def hashAndSend(self,nodeInt):
        val = self.hashNode(nodeInt)
        self.outChannels[val[0]].send(val)
        if not self.rateChannel is None:
            # print('PE'+str(charm.myPe()) + ': sending on rateChannel')
            self.rateChannel.send(1)
            # print('PE'+str(charm.myPe()) + ': Waiting on rateChannel')
            control = self.rateChannel.recv()
            # print('PE'+str(charm.myPe()) + ': Recieved on rateChannel')
            # print(control)
            while control > 0:
                control = self.rateChannel.recv()
            if control == -3:
                return False
            # print('About to leave hashAndSend')
        return True

    @coro
    def tester(self):
        print('Entered tester on PE ' + str(charm.myPe()))
        return charm.myPe()
    @coro
    def initList(self,workInts):
        self.status = Future()
        self.workInts = workInts
        self.status.send(1)

    @coro
    def initListNew(self,workInts, fut):
        self.workInts = workInts
        # print(self.workInts)
        fut.send(1)

    #@coro
    def sendAll(self,val):
        if not charm.myPe() in self.posetPElist:
            return
        for ch in self.outChannels:
            ch.send(val)
    @coro
    def resetMessageState(self):
        self.rateChannelDone = False
    @coro
    def flushMessages(self):
        if not charm.myPe() in self.overlapPElist:
            return
        self.rateChannel.send(2)
    # @coro
    # def computeSuccessors(self, callback):
    #     if len(self.workInts) > 0:
    #         successorList = [None] * len(self.workInts)
    #         for ii in range(len(successorList)):
    #             successorList[ii] = self.processNodeSuccessors(self.workInts[ii],self.N,self.constraints)
    #     else:
    #         successorList = [[set([]),-1]]


    #     self.workInts = [successorList[ii][1] for ii in range(len(successorList))]
    #     successorList = [successorList[ii][0] for ii in range(len(successorList))]
    #     # if charm.myPe() == 0:
    #     #     print('Now I got here!')
    #     self.reduce(callback, set([]).union(*successorList), Reducer.Union)
    
    @coro
    def computeSuccessorsNew(self):
        term = False
        if len(self.workInts) > 0:
            successorList = [None] * len(self.workInts)
            for ii in range(len(successorList)):
                successorList[ii] = self.processNodeSuccessors(self.workInts[ii],self.N,self.constraints)
                if successorList[ii][1] < 0:
                    term = True
                    break
        else:
            successorList = [[set([]),-1]]
        
        self.sendAll(-2 if not term else -3)

        
        self.workInts = [successorList[ii][1] for ii in range(len(successorList))]
        successorList = [successorList[ii][0] for ii in range(len(successorList))]


    @coro
    def retrieveFaces(self):
        return self.workInts

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



    def concreteMinHRep(self,H2,cnt=None,randomize=False,copyMat=True,solver='glpk',safe=False):
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
        
        # if solver=='clp':
        #     s = CyClpSimplex()
        #     xVar = s.addVariable('x', d)
        #     s.logLevel = 0
        #lp = encapsulateLP(solver, opts={'dim':d})

        idx = 0
        loc = 0
        e = np.zeros((len(H),1))
        to_keep = list(range(len(H)))
        while idx < len(H) and cntr > 0:
            e[idx,0] = 1        
            if safe:
                status, x = self.lp.runLP( \
                        H[idx,1:], \
                        -H[to_keep,1:], \
                        H[to_keep,0]+e[to_keep,0], \
                        lpopts = {'solver':solver, 'fallback':'glpk'} if solver != 'glpk' else {'solver':'glpk'}, \
                        msgID = str(charm.myPe()) \
                    )
            else:
                status, x = self.lp.runLP( \
                        H[idx,1:], \
                        -np.vstack([H[to_keep,1:], [-H[idx,1:]]]), \
                        np.hstack([H[to_keep,0]+e[to_keep,0], [-H[idx,0]]]), \
                        lpopts = {'solver':solver, 'fallback':'glpk'} if solver != 'glpk' else {'solver':'glpk'}, \
                        msgID = str(charm.myPe()) \
                    )
            e[idx,0] = 0
            
            if status != 'optimal' and (safe or status != 'primal infeasible') and status != 'dual infeasible':
                print('********************  PE' + str(charm.myPe()) + ' WARNING!!  ********************')
                print('PE' + str(charm.myPe()) + ': Infeasible or numerical ill-conditioning detected at node' )
                print('PE ' + str(charm.myPe()) + ': RESULTS MAY NOT BE ACCURATE!!')
                return [set([]), 0]
            if (safe and -H[idx,1:]@x < H[idx,0]) \
                or (not safe and (status == 'primal infeasible' or np.all(-H[to_keep,1:]@x - H[to_keep,0].reshape((len(to_keep),1)) <= 1e-10))):
                # inequality is redundant, so remove it
                to_keep.pop(loc)
                cntr -= 1
            else:
                loc += 1
                cntr -= 1
            idx += 1
        return to_keep[0:min(loc if not cnt is None else len(to_keep),len(to_keep))]

    @coro
    def processNodeSuccessorsFastLP(self,INTrep,N,H2,solver='glpk',findAll=False):
        
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
        
        to_keep_sub = self.concreteMinHRep(H[to_keep,:],cnt=findSize,copyMat=False,solver=solver)
        if findSize is None:
            findSize = len(to_keep)
        to_keep_faces = to_keep[to_keep_sub].tolist() + to_keep[findSize:len(to_keep)].tolist()
        to_keep = to_keep[to_keep_sub].tolist()

        if not findAll:
            to_keep = reorder[to_keep].tolist()
            to_keep_faces = reorder[to_keep_faces].tolist()


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
                nNode = INTrep + idx
                successors.append( \
                        nNode \
                    )
                # print('Processing node!')
                cont = self.thisProxy[self.thisIndex].hashAndSend(nNode,ret=True).get()
                # print(cont)
                if not cont:
                    return [set(successors), -1]
        
        return [set(successors), facesInt]            


        



def Union(contribs):
    return set().union(*contribs)

Reducer.addReducer(Union)



class flipConstraints:

    def __init__(self, nA, nb, pt, fA=None, fb=None):
        v = nA @ pt
        v = v.flatten() - nb.flatten()
        self.flipMapN = np.where(v<0,-1,1)
        self.flipMapSet = frozenset(np.nonzero(self.flipMapN < 0)[0])
        self.nA = np.diag(self.flipMapN) @ nA
        self.nb = np.diag(self.flipMapN) @ nb
        self.N = len(nA)
        self.d = len(nA[0])

        if (fA is not None) and (fb is not None):
            v = fA @ pt
            v = v.flatten() - fb.flatten()
            if len(np.flatnonzero(v<0)) > 0:
                raise ValueError('Supplied point must satisfy all specified \'fixed\' constraints -- i.e. fA @ pt >= fb !')
            self.fA = fA
            self.fb = fb
            self.constraints = np.vstack( ( np.hstack((-1*self.nb,self.nA)), np.hstack((-1*self.fb,self.fA)) ) )

        else:
            self.fA = None
            self.fb = None
            self.constraints = np.hstack((-self.nb,self.nA))

        self.root = 0


class flipConstraintsReduced(flipConstraints):

    def __init__(self, nA, nb, pt, fA=None, fb=None):
        super().__init__(nA, nb, pt, fA=fA, fb=fb)
        if self.fA is None:
            return
        
        # Now let's remove any hyperplanes that don't intersect the polytope defined by fA and fb
        # First let's find the vertices of the constraint polytope using CDD
        _, _, vRep = createCDDrep(self.fA, self.fb)

        redundantHyperplanes = \
            -2*np.logical_or( \
                np.all(self.nA @ vRep.T > self.nb.reshape(-1,1), axis=1), \
                np.all(self.nA @ vRep.T < self.nb.reshape(-1,1), axis=1) \
            )+1

        self.nA = np.diag(redundantHyperplanes) @ self.nA
        self.nb = np.diag(redundantHyperplanes) @ self.nb
        self.constraints = np.vstack( ( np.hstack((-1*self.nb,self.nA)), np.hstack((-1*self.fb,self.fA)) ) )
        
        # Modify root node:
        for k in np.nonzero(redundantHyperplanes<0)[0]:
            self.root += 1 << int(k)

        #print(self.root)


# Helper functions:
def createCDDrep(inputConstraintsA, inputConstraintsb):
    
    inputMat = cdd.Matrix(np.hstack((-1*inputConstraintsb,inputConstraintsA)))
    inputMat.rep_type = cdd.RepType.INEQUALITY
    inputPolytope = cdd.Polyhedron(inputMat)
    vrep = np.array(inputPolytope.get_generators())
    if len(vrep) == 0:
        raise ValueError('No vertices for input constraint polyhedron!')
    
    if np.sum(vrep[:,0]) < len(vrep):
        raise ValueError('Input constraints do not specify a closed, bounded polyhedron!')
    inputVrep = vrep[:,1:]

    return inputMat, inputPolytope, inputVrep


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