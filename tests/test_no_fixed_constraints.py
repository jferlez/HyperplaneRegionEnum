"""
Test (no charm4py) of flipConstraints objects WITHOUT fixed constraints (fA=None).

The constructor read self.hyperSet.uniqueRowIdx on that path; the attribute of a vectorSet is uniqRowIdx, so every
construction without fixed constraints raised an AttributeError.

Checked, on random arrangements that contain duplicate, negated and rescaled hyperplanes:
  1. the constructor runs, for flipConstraints and flipConstraintsReducedMin;
  2. the object agrees with one that is given the same hyperplanes plus fixed constraints that are far away from
     everything (a large box, which no hyperplane coincides with): the same non-redundant hyperplanes, the same
     flip map and redundant flips, and the same constraints for the hyperplanes;
  3. the non-redundant hyperplanes are one representative per distinct hyperplane;
  4. the object can be used: region constraints and an interior point of the root region, filterParallel,
     insertHyperplane (a new hyperplane and a duplicate), removeHyperplanes (a hyperplane with and without a
     duplicate), a pickle round trip.

Usage: python3 tests/test_no_fixed_constraints.py
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import region_helpers

failures = []
def check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}' + ('' if cond or detail == '' else f'   [{detail}]'))
    if not cond:
        failures.append(name)

rng = np.random.default_rng(0)

def randomArrangement(n, d):
    A = rng.standard_normal((n, d)); b = rng.uniform(0.3, 1.5, size=(n, 1)) * rng.choice([-1.0, 1.0], size=(n, 1))
    # copies of earlier hyperplanes: the same row, its negative, a multiple of it
    for i in range(1, n):
        k = rng.integers(6)
        if k < 3:
            j = int(rng.integers(i)); s = [1.0, -1.0, 2.5][k]
            A[i] = s * A[j]; b[i] = s * b[j]
    return A, b

def sameHyperplane(u, v, tol=1e-9):
    u = u / np.linalg.norm(u[1:]); v = v / np.linalg.norm(v[1:])
    return np.allclose(u, v, atol=tol) or np.allclose(u, -v, atol=tol)

nCases = 0; bad = []
for case in range(300):
    d = int(rng.integers(2, 5)); n = int(rng.integers(1, 9))
    A, b = randomArrangement(n, d); pt = np.zeros((d, 1))
    try:
        obj = region_helpers.flipConstraints(A.copy(), b.copy(), pt.copy())
        objMin = region_helpers.flipConstraintsReducedMin(A.copy(), b.copy(), pt.copy())
    except Exception as e:
        bad.append((case, 'constructor', f'{type(e).__name__}: {e}')); continue
    nCases += 1
    errs = []
    # the same hyperplanes inside a box that is far away from all of them
    fA = np.vstack((np.eye(d), -np.eye(d))); fb = -1.0e4 * np.ones((2 * d, 1))
    ref = region_helpers.flipConstraints(A.copy(), b.copy(), pt.copy(), fA=fA, fb=fb)
    if obj.fSet is not None or obj.fA is not None: errs.append('fSet/fA not None')
    if obj.N != ref.N or obj.allN != ref.allN or not np.array_equal(obj.nonRedundantHyperplanes, ref.nonRedundantHyperplanes): errs.append('non-redundant hyperplanes differ from the boxed object')
    elif not np.array_equal(obj.flipMapN, ref.flipMapN) or not np.array_equal(obj.redundantFlips, ref.redundantFlips): errs.append('flip map differs from the boxed object')
    elif not np.array_equal(obj.constraints[:obj.N], ref.constraints[:ref.N]) or obj.constraints.shape[0] != obj.N: errs.append('constraints differ from the boxed object')
    # one representative per distinct hyperplane
    rows = np.hstack((-b, A)); reps = obj.nonRedundantHyperplanes.tolist()
    if any(sameHyperplane(rows[i], rows[j]) for i in reps for j in reps if i < j): errs.append('two representatives of one hyperplane')
    if any(not any(sameHyperplane(rows[i], rows[j]) for j in reps) for i in range(n)): errs.append('a hyperplane without a representative')
    if objMin.fSet is not None or objMin.N > obj.N: errs.append('flipConstraintsReducedMin')
    if errs: bad.append((case, errs))
check(f'{nCases} of 300 random arrangements without fixed constraints: constructed, and the same hyperplane data as with a far-away box', nCases == 300 and len(bad) == 0, str(bad[:3]))

# ---- using such an object
nA = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]); nb = np.array([[0.5], [0.5], [1.0], [3.0], [-3.0]])
pt = np.zeros((2, 1))
obj = region_helpers.flipConstraints(nA.copy(), nb.copy(), pt.copy())
check('5 hyperplanes of which 3 are distinct: N = 3 and allN = 5', obj.N == 3 and obj.allN == 5, f'{obj.N} {obj.allN}')
H = obj.getRegionConstraints(tuple())
x = obj.regionInteriorPoint(tuple())
check('the root region has constraints for every hyperplane, and an interior point that satisfies them', \
        H.shape == (5, 3) and x is not None and bool(np.all(H[:, 1:] @ x.reshape(-1, 1) + H[:, [0]] > 0)), f'{H.shape} {None if x is None else x.flatten()}')
par = obj.filterParallel(np.array([[-0.5, 1.0, 0.0]]))
new = obj.filterParallel(np.array([[0.2, 1.0, -1.0]]))
check('filterParallel finds an existing hyperplane and nothing for a new direction', len(par[0]) == 1 and len(new[0]) == 0 and len(new[1]) == 0, f'{par} {new}')
r1 = obj.insertHyperplane(np.array([[1.0, -1.0]]), np.array([[0.2]])); n1 = (obj.N, obj.allN)
r2 = obj.insertHyperplane(np.array([[0.0, 3.0]]), np.array([[1.5]])); n2 = (obj.N, obj.allN)
check('insertHyperplane: a new hyperplane is added (N 4, allN 6), a duplicate only counts in allN (N 4, allN 7)', r1 is True and r2 is False and n1 == (4, 6) and n2 == (4, 7), f'{r1} {n1} {r2} {n2}')
# (hyperplane 5 is the one inserted above and has no duplicate; hyperplane 0 has one, which takes its place)
obj.removeHyperplanes([5], allN=True); n3 = (obj.N, obj.allN)
obj.removeHyperplanes([0], allN=True); n4 = (obj.N, obj.allN)
check('removeHyperplanes: without a duplicate N drops (3, 6); with one, the duplicate takes its place (3, 5)', n3 == (3, 6) and n4 == (3, 5) and obj.fSet is None, f'{n3} {n4}')
obj2 = pickle.loads(pickle.dumps(obj.serialize())); obj2.deserialize(); obj.deserialize()
check('pickle round trip keeps the object', obj2.fSet is None and obj2.N == obj.N and obj2.allN == obj.allN and np.array_equal(obj2.constraints, obj.constraints) \
        and np.array_equal(obj2.nonRedundantHyperplanes, obj.nonRedundantHyperplanes))

print(f'\n{"ALL TESTS PASSED" if len(failures) == 0 else str(len(failures)) + " TEST(S) FAILED"}')
sys.exit(0 if len(failures) == 0 else 1)
