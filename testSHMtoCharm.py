import multiprocessing
from multiprocessing import shared_memory
from queue import Empty
from charm4py import charm
import time
import pickle
import sys

def dist(args):
    print(args)
    shm_b = shared_memory.SharedMemory(args[1])
    buffer = shm_b.buf
    print(buffer[0])
    buffer[0]=5
    print('Num PEs: ' + str(charm.numPes()))
    print('Num hosts: ' + str(charm.numHosts()))
    shm_b.close()
    time.sleep(20)
    charm.exit()

def sched(name):
    sys.argv.append(name)
    charm.start(dist)


if __name__=='__main__':

    shm_a = shared_memory.SharedMemory(create=True, size=1)
    buffer = shm_a.buf
    buffer[0] = 100
    print(shm_a.name)
    p = multiprocessing.Process(target=sched,args=(shm_a.name,))

    p.start()

    p.join()

    print(buffer[0])

    shm_a.close()
    shm_a.unlink()

    

