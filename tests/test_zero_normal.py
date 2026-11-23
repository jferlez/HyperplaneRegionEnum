"""
Test (no charm4py) of what flipConstraints does with a "hyperplane" whose normal vector is zero, i.e. with a
constant constraint "c >= 0" (such rows come from constant functions, e.g. the level set {l(x) = bound} of a
constant local linear function of a TLL).

With normalization, the row used to be divided by ||normal|| = 0 and stayed in the object as [inf, nan, ..., nan]:
flipConstraints was created without complaint and failed in some later LP, flipConstraintsReducedMin failed in
its redundancy check with a message about the LP solver's input. Now

  1. the constructors and insertHyperplane raise a ValueError that names the cause and the offending rows, for
     the hyperplanes as well as for the fixed constraints;
  2. it is NOT a DegenerateHyperplane (callers catch that one to skip a projection);
  3. an insertion that is refused leaves the object unchanged, and usable;
  4. without normalization (normalize=None) zero normals are accepted as before: the TLL compiler creates such
     objects (the difference of two parallel local linear functions is constant), and flipConstraintsReducedMin
     then classifies the row as redundant and keeps its side in the flip map.

Usage: python3 tests/test_zero_normal.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import region_helpers

failures = []
def check(name, cond):
    print(f'{"PASS" if cond else "FAIL"}  {name}')
    if not cond:
        failures.append(name)

def raises(fn):
    try:
        fn()
    except region_helpers.DegenerateHyperplane as e:
        return 'DegenerateHyperplane', str(e)
    except ValueError as e:
        return 'ValueError', str(e)
    except Exception as e:
        return type(e).__name__, str(e)
    return None, ''

# rows "A x >= b":  x0 >= -0.9,  0 >= -1.4 (constant, true),  x1 >= -0.9,  -x0 >= -0.9
nA = np.array([[1., 0.], [0., 0.], [0., 1.], [-1., 0.]]); nb = np.array([[-0.9], [-1.4], [-0.9], [-0.9]])
fA = np.vstack([np.eye(2), -np.eye(2)]); fb = -np.ones((4, 1)); pt = np.zeros((2, 1))
keep = [0, 2, 3]

for cls in (region_helpers.flipConstraints, region_helpers.flipConstraintsReducedMin):
    kind, msg = raises(lambda: cls(nA, nb, pt, fA, fb, normalize=1.0))
    check(f'{cls.__name__}: zero normal with normalization raises ValueError', kind == 'ValueError')
    check(f'{cls.__name__}: the message names the cause and the row', 'zero' in msg and 'hyperplane(s) [1]' in msg and 'normalize=None' in msg)
    kind, msg = raises(lambda: cls(nA[keep], nb[keep], pt, np.vstack([fA, np.zeros((1, 2))]), np.vstack([fb, [[-1.0]]]), normalize=1.0))
    check(f'{cls.__name__}: zero normal among the fixed constraints raises ValueError', kind == 'ValueError' and 'fixed constraint(s) [4]' in msg)
    kind, msg = raises(lambda: cls(nA, nb, pt, normalize=1.0))
    check(f'{cls.__name__}: ... also without fixed constraints', kind == 'ValueError')
    kind, msg = raises(lambda: cls(nA[keep], nb[keep], pt, fA, fb, normalize=1.0))
    check(f'{cls.__name__}: non-degenerate rows are accepted', kind is None)

    obj = cls(nA[keep], nb[keep], pt, fA, fb, normalize=1.0)
    before = (obj.N, obj.allN, obj.allConstraints.copy(), obj.flipMapN.copy(), obj.nonRedundantHyperplanes.copy())
    kind, msg = raises(lambda: obj.insertHyperplane(np.zeros((2,)), np.array([0.3])))
    check(f'{cls.__name__}: inserting a zero normal raises ValueError', kind == 'ValueError' and 'inserted hyperplane' in msg)
    after = (obj.N, obj.allN, obj.allConstraints, obj.flipMapN, obj.nonRedundantHyperplanes)
    check(f'{cls.__name__}: the object is unchanged after the refused insertion', \
            before[:2] == after[:2] and all(np.array_equal(x, y) for x, y in zip(before[2:], after[2:])))
    check(f'{cls.__name__}: and an ordinary insertion still works', obj.insertHyperplane(np.array([1.0, 1.0]), np.array([0.3])) and obj.N == before[0] + 1)

    kind, msg = raises(lambda: cls(nA, nb, pt, fA, fb, normalize=None))
    check(f'{cls.__name__}: zero normal without normalization is accepted', kind is None)

# without normalization the redundancy filter drops the constant row and remembers its side
for offset, satisfied in ((-1.4, True), (1.4, False)):
    nb2 = nb.copy(); nb2[1, 0] = offset
    obj = region_helpers.flipConstraintsReducedMin(nA, nb2, pt, fA, fb, normalize=None)
    flags = set(int(i) for i in obj.translateRegion((), allN=True))
    check(f'normalize=None: constant row "0 >= {offset}" is redundant and reported as {"satisfied" if satisfied else "violated"}', \
            1 not in obj.nonRedundantHyperplanes.tolist() and ((1 in flags) == satisfied))

print(f'\n{"ALL TESTS PASSED" if len(failures) == 0 else str(len(failures)) + " TEST(S) FAILED"}')
sys.exit(0 if len(failures) == 0 else 1)
