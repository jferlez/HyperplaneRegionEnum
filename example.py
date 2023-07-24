from charm4py import charm, Chare
import posetFastCharm
import numpy as np


def main(args):
    # Instantiate an region enumerator:
    # (NB: args argument is required, and must be fixed length due to charm4py requirements)
    enumerator = Chare(posetFastCharm.Poset, \
                        args=[ \
                                # PE Specification (None => use all PEs)
                                None, \
                                # Constructor for individual region instances (None => posetFastCharm.PosetNode)
                                None, \
                                # localVarGroup for evaluating region tests (None => posetFastCharm.localVar)
                                None, \
                                # Chare to obtain region successors in poset (None => posetFastCharm.successorWorker)
                                None, \
                                # Execute region tests on poset workers (True) or hash workers (False)
                                False, \
                                # Specification for chaining DistributedHash
                                []
                            ], \
                        onPE=0 \
                    )
    charm.awaitCreation(enumerator)
    enumerator.init(awaitable=True).get()
    ######################################################################
    # Some example hyperplanes, defined by A @ x == b:
    A = np.array([ \
                    [1, 2], \
                    [3, 4] \
                ], dtype=np.float64)
    b = np.array([\
                    [-2], \
                    [8] \
                ],dtype=np.float64)
    
    # An initial point specifying the first region to be counted:
    pt = np.array([
                    [0], \
                    [0]
                ], dtype=np.float64)
    
    # Some fixed boundaries within which to count regions (fA @ x >= fb):
    bd = 1e6
    fA = np.array([
                    [ 1, 0], \
                    [-1, 0], \
                    [ 0,  1], \
                    [ 0, -1] \
                ],dtype=np.float64)
    fb = np.array([
                    [-bd], \
                    [-bd], \
                    [-bd], \
                    [-bd] \
                ])

    ######################################################################
    # Supply hyperplanes to enumerator (.get() call waits for completion):
    enumerator.initialize(
            [[A, b]], \
            pt, \
            fA, \
            fb, \
            awaitable=True \
        ).get()
    # Shift and pre-filter hyperplanes for intersection with fixed constraints
    enumerator.setConstraint(0, prefilter=True, awaitable=True).get()

    # Return whether the enumeration completed without the region test failing on any region:
    retVal = enumerator.populatePoset(ret=True).get()

    # Collect the stats from the enumeration:
    stats = enumerator.getStats(ret=True).get()

    print(stats)

    charm.exit()


# Currently this is required to notify charm4py of all charm-related modules
# Future versions will adapt to the new charm4py definition syntax using decorators 
charm.start(main, modules=['posetFastCharm', 'DistributedHash'] )