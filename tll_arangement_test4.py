import charm4py
from charm4py import charm, Chare, Channel, coro, Group, Future
import numpy as np
import pickle
import time
import random
import re
# import posetFastCharm
# import simple2xPosetFastCharm
import DistributedHash
import region_helpers
from simple2xSuccessorWorker import simple2xSuccessorWorker
import posetFastCharm
import posetFastCharm_numba
import TLLHypercubeReach
from copy import copy, deepcopy
import TLLnet
import os
import itertools
import vectorSet.vectorSet as vectorSet
charm.options.local_msg_buf_size = 10000

class setupCheckerVarsOriginCheck(TLLHypercubeReach.setupCheckerVarsOriginCheck, Chare):
    @coro
    def checkNodeRS(self,nodeBytes):
        retVal = True
        if type(nodeBytes) == bytearray:
            nodeBytes = tuple(posetFastCharm.bytesToList(nodeBytes,self.flippedConstraints.wholeBytes,self.flippedConstraints.tailBits))
        regSet = self.flippedConstraints.translateRegion(nodeBytes,allN=True)

        val = False
        for sSet in self.selectorSetsFull[self.out]:
            if not posetFastCharm_numba.is_non_empty_intersection(regSet,sSet):
                val = True
                break
        # print('Done check; val = ' + str(val))
        if not val:
            # This **MUST** be an ordinary method: if it's a @coro, the entire system will fail, even with suitable .get() calls
            # This behavior is totally inexplicable: for some reason, it will fail with the infamous: "No pending future with fid= ...
            # A common reason is sending to a future that already received its value(s)" message.
            retVal = False
        return retVal

tllPath = '/Users/james/Dropbox/Research/Postdoc Yasser/code/TLLVerifyBench/tll/'
class Main(Chare):
    @coro
    def __init__(self,args):

        with open('sizeVsTime_n2_input.p', 'rb') as fp:
            expers = pickle.load(fp)

        localVarGroup = Group(setupCheckerVarsOriginCheck,args=[])
        charm.awaitCreation(localVarGroup)

        poset = Chare(posetFastCharm.Poset,args=[
                # Hash lists:
                {'poset':[(0,1,1)],'hash':[(0,1,1)],'gpu':[]}, \
                # node constructor
                # TLLHypercubeReach.PosetNodeTLLVerOriginCheck, \
                None, \
                # Local variable group
                # localVarGroup, \
                None, \
                # successor worker chare
                None,
                # Use poset checking
                True, \
                # feederSpec
                [] \
        ],onPE=0)
        # poset = Chare(simple2xPosetFastCharm.PosetSimple2x,args=[],onPE=0)
        charm.awaitCreation(poset)

        poset.init( \

                awaitable=True
            ).get()
        succGroupProxy = poset.getSuccGroupProxy(ret=True).get()
        localVarGroup.init(succGroupProxy,list(range(16)))

        hashTable = poset.getHashTableProxy(ret=True).get()
        # hashTable.setCheckDispatch({'update':'foo'},awaitable=True).get()
        tabEnum = hashTable.registerEnumChannels(self.thisProxy,awaitable=True).get()

        tllList = os.listdir(tllPath)

        for sizeIdx in [2]: #range(len(expers)):
            for experIdx in [1]: #range(len(expers[sizeIdx])):
                fileList = [p for p in tllList if re.match(f'.*{sizeIdx}_{experIdx}.tll',p)]

                tll = TLLnet.TLLnet.fromTLLFormat(tllPath + fileList[0])


                # *******************************************************************************************
                # **********   Fixed Constraints consisting of a large box to capture all regions  **********
                # *******************************************************************************************
                fixedConstraintsA = []
                fixedConstraintsb = []
                # boxSize = 5
                bd = 1e6
                incDim = 1
                for i in range(tll.n+incDim):
                    fixedConstraintsA.append([1 if j == i else 0 for j in range(tll.n+incDim)])
                    fixedConstraintsA.append([-1 if j == i else 0 for j in range(tll.n+incDim)])
                    fixedConstraintsb.append(-bd)
                    fixedConstraintsb.append(-bd)
                    # fixedConstraintsb.append(boxSize)
                    # fixedConstraintsb.append(-boxSize)

                # fixedConstraintsA = np.array(fixedConstraintsA)
                fixedConstraintsb = np.array([fixedConstraintsb],dtype=np.float64).transpose()
                fixedConstraintsA = np.array(fixedConstraintsA,dtype=np.float64)
                print(f'\n\n{tll.N}\n\n')
                # testSet = vectorSet.vectorSet( np.hstack([-fixedConstraintsb,fixedConstraintsA]) )
                # test = np.hstack([-fixedConstraintsb,fixedConstraintsA])
                # print(type(testSet))
                # print(vectorSet.vecCompareNb)
                # with open('prob_mat.p','wb') as fp:
                #     pickle.dump( test, fp)
                # print(testSet.sortOrd)
                # print(f'{test.flags} {test.dtype}')
                # print(test[testSet.sortOrd,:])
                # print(test[0,:])
                # print(test[1,:])
                # print(vectorSet.vecCompareNb(test[0,:],test[1,:],1e-9,1e-9))
                # charm.exit()
                subs = tll.localLinearFns[0][0].shape[0]
                # subs = 4
                localLinearFns = [ [kernBias[0][:subs,:].copy(), kernBias[1][:subs].copy().reshape( (-1,1) )] for kernBias in tll.localLinearFns ]
                random_biases = np.hstack([localLinearFns[0][0], 2 * np.random.random_sample((localLinearFns[0][0].shape[0],incDim)) - 1 ])
                numDegen = 8
                newNormals = np.random.random_sample((numDegen,random_biases.shape[1]))
                # newNormals = np.array([ \
                #     [-0.10365656, -0.53111269], \
                #     [-0.28945417, -0.5266667 ], \
                #     [-0.47478714, -0.82734073] \
                #     ])
                # numDegen = newNormals.shape[0]

                # NUMBER OF HYPERPLANES TO INSERT
                Nrem = 10

                # Override randomized input with saved values for debugging
                with open('results_remove_1785080042_KEEP_preserve.p','rb') as fp:
                    tempDict = pickle.load(fp)
                newNormals = tempDict['newNormals']
                numDegen = tempDict['numDegen']
                Nrem = tempDict['Nrem']
                random_biases = tempDict['random_biases']


                localLinearFns[0][0] = random_biases

                # localLinearFns[0][0] = np.vstack([localLinearFns[0][0], np.random.random_sample((numDegen,2))])
                localLinearFns[0][0] = np.vstack([localLinearFns[0][0], newNormals])
                localLinearFns[0][1] = np.vstack([localLinearFns[0][1], np.zeros((numDegen,1))])
                # print(localLinearFns)
                # with open('localLinearFns.p','wb') as fp:
                #     pickle.dump(localLinearFns,fp)
                # with open('localLinearFns_d2_problem.p','rb') as fp:
                #     localLinearFns = pickle.load(fp)
                # localLinearFns[0][0] = localLinearFns[0][0][:-1,:]
                # localLinearFns[0][1] = localLinearFns[0][1][:-1,:]


                localLinearFnsRem = [ [kernBias[0].copy()[:-Nrem,:], kernBias[1].copy().reshape( (-1,1) )[:-Nrem,:]] for kernBias in localLinearFns ]
                # localLinearFns2 = [ [kernBias[0].copy()[:-2,:], kernBias[1].copy().reshape( (-1,1) )[:-2,:]] for kernBias in tll.localLinearFns ]

                # ****************************************************
                # **********   Actually do the enumeration  **********
                # ****************************************************

                pt = np.full(localLinearFns[0][0].shape[1],0,dtype=np.float64).reshape(-1,1)
                # pt = np.array([[-10.0, 234.432]],dtype=np.float64).T

                stat = poset.initialize(localLinearFns, pt, fixedConstraintsA, fixedConstraintsb, normalize=1.0, awaitable=True)
                stat.get()
                stat = poset.setConstraint(lb=0,prefilter=True,awaitable=True)
                stat.get()
                localVarGroup.initialize(tll.selectorSets)
                constraints = poset.getConstraintsObject(ret=True).get()
                constraints.insertHyperplane(-localLinearFns[0][0][0,:],localLinearFns[0][1][0,])
                constraints.insertHyperplane(-localLinearFns[0][0][0,:],localLinearFns[0][1][0,])
                constraints.insertHyperplane(constraints.constraints[-1,1:],-constraints.constraints[-1,0])

                # test = np.hstack([localLinearFns[0][1][0,] + 10.3, -2.3*localLinearFns[0][0][0,:]])
                # test = np.hstack([localLinearFns[0][1][0,], localLinearFns[0][0][0,:]])
                # print(test)
                # testPar = constraints.filterParallel(test)

                # print(testPar)
                # charm.exit()
                constraints.serialize()
                localVarGroup.setConstraint(constraints,0,awaitable=True).get()
                constraints = deepcopy(poset.getConstraintsObject(ret=True).get())

                stat = poset.initialize(localLinearFnsRem, pt, fixedConstraintsA, fixedConstraintsb, normalize=1.0, awaitable=True)
                stat.get()
                stat = poset.setConstraint(lb=0,prefilter=True,awaitable=True)
                stat.get()
                localVarGroup.initialize(tll.selectorSets)
                localVarGroup.setConstraint(poset.getConstraintsObject(ret=True).get(),0,awaitable=True).get()
                constraintsRem = deepcopy(poset.getConstraintsObject(ret=True).get())
                constraintsRem_orig = deepcopy(poset.getConstraintsObject(ret=True).get())
                for i in range(-Nrem,0):
                    print(f'  Trying to insert {i} of {Nrem}')
                    constraintsRem.insertHyperplane(-localLinearFns[0][0][i,:],localLinearFns[0][1][i])




                print(f'\n{constraintsRem.constraints}')
                print(f'\n{constraints.constraints}')
                print(f'\n***** constraints1 - constraints = {(np.allclose(constraintsRem.constraints , constraints.constraints))} *****\n')
                #poset.populatePoset(retChannelEndPoint=self.thisProxy, checkNodesFuture=checkFut, checkNodeGroup=checkerGroup, useParNodeSched=True)
                #poset.populatePoset(retChannelEndPoint=self.thisProxy, checkNodesFuture=checkFut, method='fastLP', solver='clp', findAll=True)
                print('Working on sizeIdx ' + str(sizeIdx) + ', instanceIdx ' + str(experIdx))
                t = time.time()
                optsDict = { \
                        'method':'fastLP', \
                        'solver':'glpk', \
                        'lpopts':{'tol_bnd':1e-9,'basis_fac':'luf+ft'}, \
                        'clearTable':False, \
                        'sendFaces':True, \
                        'sendWitness':True, \
                        'findAll':False, \
                        'useBounding':False, \
                        'useQuery':False, \
                        'hashStore':'bits', \
                        'useNumba':True, \
                        'minimalSimplexQtys':True, \
                        'useConeMask':False, \
                        'tol':1e-9, \
                        'rTol':1e-9, \
                        'numbaMode':'pre-compiled', \
                        'INITIAL_GPU_CHUNK':5, \
                        'GPU_CHUNK':1000, \
                        'PING_LEAD':5, \
                        'verbose':  6\
                    }
                # poset.populatePoset(opts=optsDict, awaitable=True).get() # use this call for no return channel
                poset.newTable(f'{sizeIdx}_{experIdx}_preinsert',awaitable=True).get()
                poset.activateTable(f'{sizeIdx}_{experIdx}_preinsert',awaitable=True).get()
                times = {'initial_enum_time':time.time(), 'insert_time':0, 'full_enum_time':0}
                poset.populatePoset(opts=optsDict, awaitable=True).get() # use this call for no return channel
                times['initial_enum_time'] = time.time() - times['initial_enum_time']


                hashTable = poset.getHashTableProxy(ret=True).get()
                fullTable = hashTable.getTable(ret=True).get()
                # print(list(itertools.chain.from_iterable(fullTable)))

                poset.newTable(f'{sizeIdx}_{experIdx}_insert',awaitable=True).get()
                tabStat = poset.copyTable(src=f'{sizeIdx}_{experIdx}_preinsert',dest=f'{sizeIdx}_{experIdx}_insert',ret=True).get()
                print(tabStat)
                poset.activateTable(f'{sizeIdx}_{experIdx}_insert',awaitable=True).get()
                print(f'    Active Table Idx = {hashTable.getTabIdx(ret=True).get()}')
                times['insert_time'] = time.time()
                intermediate_insert_results = []
                for i in range(-Nrem, 0):
                    newA = localLinearFns[0][0][i,:]
                    newb = localLinearFns[0][1][i]
                    if i == -4 or i == -3 or i == -2:
                        optsDict['verbose'] = 10
                    poset.insertHyperplane(newA,newb,opts=optsDict,awaitable=True).get()
                    if i == -4 or i == -3 or i == -2:
                        optsDict['verbose'] = 5
                    preserveConstraints = deepcopy(poset.getConstraintsObject(ret=True).get())
                    poset.newTable('temp_table',awaitable=True).get()
                    poset.copyTable(src=f'{sizeIdx}_{experIdx}_insert', dest='temp_table', awaitable=True).get()
                    poset.activateTable('temp_table',awaitable=True).get()
                    tempInsertTable = hashTable.getTable(ret=True).get()
                    tempInsertObj = deepcopy(poset.getConstraintsObject(ret=True).get().serialize())
                    print(f'Beginning temporary canonicalization')
                    poset.canonicalizeTable(rebasePt=constraints.pt, awaitable=True).get()
                    intermediate_insert_results.append({'table':hashTable.getTable(ret=True).get(), \
                            'insertTable':tempInsertTable, \
                            'insertConstraintsObj': tempInsertObj.serialize(), \
                            'constraintsObj':deepcopy(poset.getConstraintsObject(ret=True).get()).serialize()})
                    poset.activateTable(f'{sizeIdx}_{experIdx}_insert',awaitable=True).get()
                    poset.deleteTable('temp_table',awaitable=True).get()
                    poset.initAndSetFromConstraints(preserveConstraints,awaitable=True).get()
                times['insert_time'] = time.time() - times['insert_time']
                finalInsertConstraints = deepcopy(poset.getConstraintsObject(ret=True).get())
                # newA = tll.localLinearFns[0][0][-1,:]
                # newb = tll.localLinearFns[0][1][-1]
                # poset.insertHyperplane(newA,newb,opts=optsDict,awaitable=True).get()
                hashTable = poset.getHashTableProxy(ret=True).get()
                tempTestTable = hashTable.getTable(ret=True).get()


                poset.newTable(f'{sizeIdx}_{experIdx}',awaitable=True).get()
                poset.activateTable(f'{sizeIdx}_{experIdx}',awaitable=True).get()
                stat = poset.initialize(localLinearFns, pt, fixedConstraintsA, fixedConstraintsb, normalize=1.0, awaitable=True)
                stat.get()
                stat = poset.setConstraint(lb=0,prefilter=True,awaitable=True)
                stat.get()
                constraints = deepcopy(poset.getConstraintsObject(ret=True).get())
                times['full_enum_time'] = time.time()
                poset.populatePoset(opts=optsDict, awaitable=True).get() # use this call for no return channel
                times['full_enum_time'] = time.time() - times['full_enum_time']
                # poset.clearHashTable(awaitable=True).get()
                print(poset.getTableNames(ret=True).get())
                print('Time to enumerate regions is: ' + str(time.time()-t))

                print(f'\n\nBeginning canonicalization...\n\n')
                poset.initAndSetFromConstraints(finalInsertConstraints.serialize(),ret=True).get()
                garbage = poset.activateTable(f'{sizeIdx}_{experIdx}_insert',awaitable=True).get()

                # print(f'Pre-canonicalization length = {sum([len(x) for x in hashTable.getTable(ret=True).get()])}')
                print(f'Pre-canonicalization length = {len(hashTable.getTable(ret=True).get())}')
                print(constraints.pt)
                poset.canonicalizeTable(rebasePt=constraints.pt,awaitable=True).get()

                poset.newTable(f'{sizeIdx}_{experIdx}_remove',awaitable=True).get()
                tabStat = poset.copyTable(src=f'{sizeIdx}_{experIdx}_insert',dest=f'{sizeIdx}_{experIdx}_remove',ret=True).get()
                print(tabStat)
                poset.activateTable(f'{sizeIdx}_{experIdx}_remove',awaitable=True).get()
                print(f'    Active Table Idx (removal) = {hashTable.getTabIdx(ret=True).get()}')
                print(f'        finalInsertConstraints.N = {finalInsertConstraints.N} {-Nrem}')
                poset.removeHyperplanes([finalInsertConstraints.N+i for i in range(-Nrem, 0)],opts=optsDict,awaitable=True).get()
                finalRemoveConstraints = deepcopy(poset.getConstraintsObject(ret=True).get())

        # tabStat = poset.copyTable(src='0_1',dest='foo',ret=True).get()
        # print(tabStat)
        # print(f'Poset 1_2 length = {poset.getTableLen(ret=True).get()}')
        # tabStat = poset.activateTable('foo',awaitable=True).get()
        # print(tabStat)
        # print(f'Poset foo length = {poset.getTableLen(ret=True).get()}')
        # # poset.deleteTable('0_1',ret=True).get()
        # print(poset.getTableNames(ret=True).get())

        # f = Future()
        # hashTable.initListening(f,queryReturnInfo=True,awaitable=True).get()
        # f.get()
        # succGroupProxy.startListening(awaitable=True).get()

        # retVal = succGroupProxy[0].query([bytearray(b'\x91'),None,8],op=DistributedHash.QUERYOP_DELETE,awaitable=True).get()
        # print(retVal)
        # retVal = succGroupProxy[0].query([bytearray(b'\x91'),None,8],op=DistributedHash.QUERYOP_DELETE,awaitable=True).get()
        # print(retVal)

        # hashTable.awaitPending(usePosetChecking=False,awaitable=True).get()
        # succGroupProxy.sendAll(-2,awaitable=True).get()
        # succGroupProxy.closeQueryChannels(awaitable=True).get()
        # succGroupProxy.flushMessages(ret=True).get()


        # rdyFut = Future()
        # hashTable.enumTable(self.thisProxy,rdyFut)
        # rdyFut.get()

        # data = self.enumChannels['data']
        # ctrl = self.enumChannels['ctrl']

        # retVal = -1
        # while not retVal is None:
        #     ctrl.send(1)
        #     retVal = data.recv()
        #     if not retVal is None: print(retVal)

        poset.activateTable(f'{sizeIdx}_{experIdx}_preinsert',awaitable=True).get()
        hashTable = poset.getHashTableProxy(ret=True).get()
        fullTable = hashTable.getTable(ret=True).get()
        print(f'\n\nOriginal Poset length = {poset.getTableLen(ret=True).get()}')

        poset.activateTable(f'{sizeIdx}_{experIdx}_insert',awaitable=True).get()
        fullTableInsert = hashTable.getTable(ret=True).get()
        print(f'\n\nNew Poset length = {poset.getTableLen(ret=True).get()}')

        poset.activateTable(f'{sizeIdx}_{experIdx}_remove',awaitable=True).get()
        fullTableRemove = hashTable.getTable(ret=True).get()
        print(f'\n\nRemove Poset length = {poset.getTableLen(ret=True).get()}')

        poset.activateTable(f'{sizeIdx}_{experIdx}',awaitable=True).get()
        fullTableExpected = hashTable.getTable(ret=True).get()
        print(f'\n\nNew Poset length = {poset.getTableLen(ret=True).get()}')

        t0 = times['initial_enum_time']
        t1 = times['insert_time']
        t2 = times['full_enum_time']
        print(f'\n\nExecution times:\nInitial enumeration time: {t0}\nInsertion time: {t1}\nFull enumeration time: {t2}\n\n')

        with open(f'results_remove_{int(time.time())}.p','wb') as fp:
            pickle.dump({ \
                    'constraintsObj_expected':constraints.serialize() , \
                    'table_expected':fullTableExpected , \
                    'constraintsObj_pre':constraintsRem_orig.serialize() , \
                    'table_pre':fullTable , \
                    'constraintsObj_post':finalInsertConstraints.serialize(), \
                    'table_orig_post':fullTableInsert, \
                    'intermediate_insert_results': intermediate_insert_results, \
                    'constraintsObj_remove':finalRemoveConstraints.serialize(), \
                    'table_orig_remove':fullTableRemove, \
                    'tempTestTable': tempTestTable, \
                    'numDegen': numDegen, \
                    'Nrem': Nrem, \
                    'newNormals': newNormals, \
                    'random_biases': random_biases \
                }, \
                fp)

        charm.exit()

    @coro
    def registerEnum(self,enumPxy):
        self.enumChannels = {'data':Channel(self,remote=enumPxy),'ctrl':Channel(self,remote=enumPxy)}

charm.start(Main,modules=['posetFastCharm','DistributedHash','simple2xSuccessorWorker','TLLHypercubeReach'])
