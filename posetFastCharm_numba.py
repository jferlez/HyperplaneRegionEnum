from typing import List
import numpy as np
import numba as nb

from numba.pycc import CC


cc = CC('posetFastCharm_numba')
cc.verbose = True

# Hash function from:
# https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key

# @nb.cfunc(nb.int64(nb.byte[:] ) )
@cc.export('hashNodeBytes','uint64(uint8[:])')
def hashNodeBytes(nodeBytes):
    shift = np.left_shift(np.uint64(1),np.arange(0,64,8,dtype=np.uint64))
    chunks = np.full(len(nodeBytes)//8 + (1 if len(nodeBytes) <= 8 else 0),0,dtype=np.uint64)
    hashInt = np.full(len(chunks),0,dtype=np.uint64)
    tail = 8
    for i in range(len(chunks)):
        if i == len(chunks)-1:
            tail = len(nodeBytes) % 8
            chunks[i] = np.sum(nodeBytes[i:i+tail] * shift[0:tail],dtype=np.uint64)
        else:
            chunks[i] = np.sum(nodeBytes[i:i+tail] * shift[0:tail],dtype=np.uint64)

    p = np.right_shift(chunks,np.uint64(32))
    p = np.uint64(6148914691236517205) * np.bitwise_xor(chunks, p)
    hashInt = np.uint64(17316035218449499591) * np.bitwise_xor(p , np.right_shift(p,np.uint64(32)))
    
    for i in range(1, len(hashInt)):
        hashInt[0] = np.bitwise_xor(hashInt[0] , hashInt[i])
    return hashInt[0]

# Equivalent pure python version
# def hashNodeBytes(nodeBytes):
#     chunks = np.array( \
#         [int.from_bytes(nodeBytes[idx:min(idx+8,len(nodeBytes))],'little') for idx in range(0,len(nodeBytes),8)], \
#         dtype=np.uint64 \
#         )
#     p = np.uint64(6148914691236517205) * np.bitwise_xor(chunks, np.right_shift(chunks,np.uint64(32)))
#     hashInt = np.uint64(17316035218449499591) * np.bitwise_xor(p, np.right_shift(p,np.uint64(32)))
#     return int(np.bitwise_xor.reduce(hashInt))

# @cc.export('is_in_set_idx','int64[:](int64[:],int64[:])')
@cc.export('is_in_set_idx',nb.int64[:](nb.int64[:],nb.types.List(nb.int64)))
def is_in_set_idx(a, b):
    a = a.ravel()
    n = len(a)
    n = len(a)
    result = np.full(n, 0)
    set_b = set(b)
    idx = 0
    for i in range(n):
        if a[i] in set_b:
            result[idx] = i
            idx += 1
    return result[0:idx].flatten()


# @nb.cfunc(nb.types.boolean(nb.int64[:],nb.types.Set(nb.int64, reflected=True)) )
@cc.export('is_non_empty_intersection','boolean(int64[:],Set(int64))')
def is_non_empty_intersection(a, set_b):
    retVal = False
    a = a.ravel()
    n = len(a)
    # set_b = set(b)
    for i in range(n):
        if a[i] in set_b:
            retVal = True
            return retVal
    return retVal


if __name__=='__main__':
    cc.compile()