import encapsulateLP
import numpy as np
import region_helpers
import pickle
with open('lpParams_1785130605.1460018_0_KEEP.p','rb') as fp:
    lparams = pickle.load(fp)
H = lparams['H']
H2 = lparams['H2']
offsetIdx = lparams['offsetIdx']
intIdx = lparams['intIdx']
idx = lparams['idx']
lpopts = lparams['lpopts']
constraint_list = lparams['constraint_list']
lpObj = encapsulateLP.encapsulateLP()
H[offsetIdx,0] += 1
status, x = lpObj.runLP( \
                         H2[intIdx[idx],1:], \
                         -H[constraint_list,1:], \
                         H[constraint_list,0], \
                         lpopts =lpopts, \
                         msgID = str(0) \
                     )
# result is primal infeasible on ARM64 and optimal on AMD64