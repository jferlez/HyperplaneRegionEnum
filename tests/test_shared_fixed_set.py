"""
Test (no charm4py) of the fSet argument of the flipConstraints constructor: an object that is given the vectorSet
of its fixed constraints must be the same object as one that builds that set itself.

Checked, on random arrangements with fixed constraints (including hyperplanes that coincide with a fixed
constraint, duplicate hyperplanes, no hyperplanes at all, and an anchor point on a hyperplane):
  * every array and index of the two objects is equal (constraints, allConstraints, nonRedundantHyperplanes, N,
    allN, norms, flip map, anchor point, redundantFlips);
  * filterParallel and projectConstraints give the same answers for random query hyperplanes, and for a query
    that coincides with a fixed constraint;
  * several objects with the same fixed constraints share ONE set, and using them does not change it;
  * a set of the wrong size, or something that is not a vectorSet, raises a ValueError;
  * the time of the constructor with and without the argument (reported, not checked).

Usage: python3 tests/test_shared_fixed_set.py
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import region_helpers
from region_helpers import flipConstraints
vectorSet = region_helpers.vectorSet

failures = []
def check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}{("   " + str(detail)) if detail != "" and not cond else ""}', flush=True)
    if not cond:
        failures.append(name)

rng = np.random.default_rng(0)

def randomProblem(d, m, k):
    # fixed constraints: m random half-spaces around a point q, plus a box; k hyperplanes, some special
    q = rng.uniform(-1, 1, size=(d, 1))
    A = rng.normal(size=(m, d)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    b = A @ q - np.abs(rng.normal(scale=0.5, size=(m, 1))) - 0.05                  # A q >= b
    fA = np.vstack([A, np.eye(d), -np.eye(d)]); fb = np.vstack([b, -3.0 * np.ones((2 * d, 1))])
    nA = rng.normal(size=(k, d)); nb = nA @ q + rng.normal(scale=0.3, size=(k, 1))
    for i in range(k):
        kind = rng.integers(0, 8)
        if kind == 0 and m > 0:      # coincides with a fixed constraint (possibly rescaled / negated)
            j = rng.integers(0, fA.shape[0]); s = rng.choice([-2.0, 0.5, 1.0]); nA[i] = s * fA[j]; nb[i] = s * fb[j]
        elif kind == 1 and i > 0:    # duplicate of an earlier hyperplane
            j = rng.integers(0, i); s = rng.choice([-1.0, 3.0]); nA[i] = s * nA[j]; nb[i] = s * nb[j]
        elif kind == 2:              # through the anchor point
            nb[i] = nA[i] @ q
    return nA, nb, q, fA, fb

def sameObject(a, b):
    problems = []
    for attr in ('constraints', 'allConstraints', 'nonRedundantHyperplanes', 'nrms', 'allNrms', 'flipMapN', 'pt', 'redundantFlips', 'nA', 'nb'):
        if not np.array_equal(np.asarray(getattr(a, attr)), np.asarray(getattr(b, attr))): problems.append(attr)
    for attr in ('N', 'allN', 'd', 'wholeBytes', 'tailBits', 'wholeBytesAllN', 'tailBitsAllN'):
        if getattr(a, attr) != getattr(b, attr): problems.append(attr)
    if a.hyperSet.uniqRowIdx != b.hyperSet.uniqRowIdx: problems.append('hyperSet.uniqRowIdx')
    return problems

nObj = 0; nDiff = 0; nQueryDiff = 0; details = []
for case in range(600):
    d = int(rng.integers(1, 6)); m = int(rng.integers(0, 25)); k = int(rng.choice([0, 1, 2, 3, 6]))
    nA, nb, q, fA, fb = randomProblem(d, m, k)
    shared = vectorSet.vectorSet(np.hstack((-1 * fb, fA)))
    a = flipConstraints(nA, nb, q, fA=fA, fb=fb)
    b = flipConstraints(nA, nb, q, fA=fA, fb=fb, fSet=shared)
    nObj += 1
    p = sameObject(a, b)
    if p: nDiff += 1; details.append((case, p))
    if b.fSet is not shared: nDiff += 1; details.append((case, 'set was copied'))
    # queries that read the fixed set
    for trial in range(4):
        if trial == 0 and fA.shape[0] > 0:
            j = rng.integers(0, fA.shape[0]); h = np.hstack([-fb[j], fA[j]]) * rng.choice([-1.5, 1.0])
        else:
            h = np.hstack([rng.normal(), rng.normal(size=d)])
        fa, fb_ = a.filterParallel(h), b.filterParallel(h)
        same = all(list(x) == list(y) for x, y in zip(fa, fb_))
        if d > 1:
            try: pa = a.projectConstraints(h)
            except region_helpers.DegenerateHyperplane: pa = 'degenerate'
            try: pb = b.projectConstraints(h)
            except region_helpers.DegenerateHyperplane: pb = 'degenerate'
            if (pa is None) != (pb is None) or isinstance(pa, str) != isinstance(pb, str): same = False
            elif pa is not None and not isinstance(pa, str):
                same = same and np.array_equal(pa[0], pb[0]) and pa[1] == pb[1] and list(pa[2]) == list(pb[2])
        nQueryDiff += (not same)
check(f'{nObj} random objects: identical with and without the fSet argument, and the given set is used as it is', nDiff == 0, details[:3])
check('filterParallel and projectConstraints give the same answers on both', nQueryDiff == 0, f'{nQueryDiff} differ')

# several objects on one set; using them leaves the set as it was
d, m = 3, 20
nA, nb, q, fA, fb = randomProblem(d, m, 0)
shared = vectorSet.vectorSet(np.hstack((-1 * fb, fA)))
before = (list(shared.uniqRowIdx), shared.Nunique, np.array(shared.getUniqueRows()), shared.N)
objs = []
for i in range(5):
    a_ = rng.normal(size=(i, d))
    objs.append(flipConstraints(a_, a_ @ q - 0.2, q, fA=fA, fb=fb, fSet=shared))
for o in objs:
    o.filterParallel(np.hstack([rng.normal(), rng.normal(size=d)])); o.getRegionConstraints(tuple())
    o.projectConstraints(np.hstack([-(rng.normal(size=d) @ q.flatten()), rng.normal(size=d)]))
check('five objects share one set', all(o.fSet is shared for o in objs))
check('the shared set is unchanged after they were used', list(shared.uniqRowIdx) == before[0] and shared.Nunique == before[1] and np.array_equal(shared.getUniqueRows(), before[2]) and shared.N == before[3])

for label, bad in (('a set with another number of rows', vectorSet.vectorSet(np.hstack((-1 * fb, fA))[:-1, :])), ('a set of another dimension', vectorSet.vectorSet(np.hstack((-1 * fb, fA, fA[:, :1])))), ('something that is not a vectorSet', np.hstack((-1 * fb, fA)))):
    try:
        flipConstraints(np.ones((1, d)), np.array([[5.0]]), q, fA=fA, fb=fb, fSet=bad)
        check(f'{label} raises ValueError', False)
    except ValueError:
        check(f'{label} raises ValueError', True)

print()
for (m, d) in [(20, 3), (60, 3), (150, 5)]:
    probs = [randomProblem(d, m, 1) for _ in range(200)]
    sets = [vectorSet.vectorSet(np.hstack((-1 * p[4], p[3]))) for p in probs]
    t = time.perf_counter()
    for p in probs: flipConstraints(p[0], p[1], p[2], fA=p[3], fb=p[4])
    t1 = (time.perf_counter() - t) / len(probs)
    t = time.perf_counter()
    for p, sv in zip(probs, sets): flipConstraints(p[0], p[1], p[2], fA=p[3], fb=p[4], fSet=sv)
    t2 = (time.perf_counter() - t) / len(probs)
    print(f'   {m + 2 * d:4d} fixed rows, {d} dimensions: constructor {1e3 * t1:.3f} ms without the argument, {1e3 * t2:.3f} ms with it')

print(f'\n{"ALL TESTS PASSED" if len(failures) == 0 else str(len(failures)) + " TEST(S) FAILED"}')
sys.exit(0 if len(failures) == 0 else 1)
