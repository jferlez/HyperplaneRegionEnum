"""
Test (no charm4py) of what flipConstraints guarantees when a hyperplane passes exactly through the anchor point,
and of the translation of region labels between objects with different anchor points (setRebase/rebaseRegion).

A region is labeled by the hyperplanes that separate it from the anchor point of the object. Code that carries a
label from one object to another (e.g. the table merges of the TLL compiler) used to assume that both objects have
their anchor in the same region, which is true as long as they were all created with the same anchor point and no
hyperplane passes through it. For a cube and hyperplanes like x0 = x4 that assumption fails: the anchor (the centre
of the cube) lies on the hyperplane, each object picks a side, and moves its own anchor there.

Checked here, for random arrangements and for arrangements with hyperplanes through the anchor:
  1. tie rule: a single hyperplane through the anchor keeps the orientation it was given, both in the constructor
     and in insertHyperplane;
  2. whatever the number of hyperplanes through the anchor, construction/insertion succeeds, and afterwards the
     anchor is strictly inside the root region (the region with the empty label). (With several hyperplanes
     through one point, the sides chosen by the tie rule need not intersect, e.g. three lines in the plane.)
  3. the label of a point, computed from the signs of the constraints, is the set of hyperplanes that separate
     the point from the anchor;
  4. for two objects A and B with the same hyperplanes but different anchors (and therefore different
     orientations and labels), A.setRebase(B.pt) followed by A.rebaseRegion(label in B) is the label in A.

Usage: python3 tests/test_anchor_ties.py [numSeeds]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import region_helpers

failures = []
def check(name, cond):
    if not cond:
        failures.append(name)
        print(f'FAIL  {name}')

def label(obj, x):
    # Region label of the point x: the (non-redundant) hyperplanes on whose negative side x lies
    v = obj.constraints[:obj.N, 0] + obj.constraints[:obj.N, 1:] @ x.flatten()
    return tuple(int(i) for i in np.nonzero(v < 0)[0])

def separating(obj, x, y):
    vx = obj.constraints[:obj.N, 0] + obj.constraints[:obj.N, 1:] @ x.flatten()
    vy = obj.constraints[:obj.N, 0] + obj.constraints[:obj.N, 1:] @ y.flatten()
    return tuple(int(i) for i in np.nonzero(vx * vy < 0)[0])

def sameSide(row, a, c0):
    # True if the stored row [c0', a'] is a positive multiple of the given [c0, a]
    k = int(np.argmax(np.abs(a)))
    return (row[1 + k] > 0) == (a[k] > 0)

numSeeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for cls in [region_helpers.flipConstraints, region_helpers.flipConstraintsReducedMin]:
    for seed in range(numSeeds):
        rng = np.random.default_rng(seed)
        d = int(rng.integers(2, 6))
        fA = np.vstack([np.eye(d), -np.eye(d)])
        fb = -np.ones((2 * d, 1))                       # the cube [-1, 1]^d
        centre = np.zeros((d, 1))
        nGen, nTie = int(rng.integers(1, 5)), int(rng.integers(1, 6))
        A = rng.normal(size=(nGen + nTie, d))
        b = np.vstack([0.4 * rng.normal(size=(nGen, 1)), np.zeros((nTie, 1))])     # the last nTie pass through the centre
        tag = f'{cls.__name__} seed {seed}'

        # ---- hyperplanes through the anchor given to the constructor ----
        obj = cls(A.copy(), b.copy(), centre.copy(), fA=fA, fb=fb)
        allRows = obj.allConstraints[:obj.allN]
        if nTie == 1:
            check(f'{tag}: constructor keeps the orientation of a hyperplane through the anchor', sameSide(allRows[nGen], A[nGen], -b[nGen, 0]))
        v = obj.constraints[:obj.N, 0] + obj.constraints[:obj.N, 1:] @ obj.pt.flatten()
        check(f'{tag}: the anchor is strictly inside the root region', obj.N == 0 or float(np.min(v)) > obj.tol)
        check(f'{tag}: the anchor was moved off the hyperplanes', not np.array_equal(obj.pt, centre))

        # ---- the same hyperplanes inserted one at a time into an object without hyperplanes ----
        ins = cls(np.ones((0, d)), np.ones((0, 1)), centre.copy(), fA=fA, fb=fb)
        for i in range(nGen + nTie):
            ins.insertHyperplane(A[i].copy(), b[i].copy())
        insRows = ins.allConstraints[:ins.allN]
        # (only the first of them is a tie: it moves the anchor off the centre)
        check(f'{tag}: insertHyperplane keeps the orientation of a hyperplane through the anchor', sameSide(insRows[nGen], A[nGen], -b[nGen, 0]))
        v = ins.constraints[:ins.N, 0] + ins.constraints[:ins.N, 1:] @ ins.pt.flatten()
        check(f'{tag}: the anchor is strictly inside the root region after insertions', ins.N == 0 or float(np.min(v)) > ins.tol)

        # ---- an object for the same arrangement with its anchor in another region ----
        other = None
        for _ in range(50):
            cand = rng.uniform(-0.9, 0.9, size=(d, 1))
            vals = A @ cand - b
            if np.min(np.abs(vals)) > 1e-3 and len(separating(obj, cand, obj.pt)) > 0:
                other = cls(A.copy(), b.copy(), cand.copy(), fA=fA, fb=fb)
                break
        if other is None or other.N != obj.N or not np.array_equal(other.nonRedundantHyperplanes, obj.nonRedundantHyperplanes):
            continue
        check(f'{tag}: an anchor off all hyperplanes is not moved', np.array_equal(other.pt, cand))
        obj.setRebase(other.pt)
        other.setRebase(obj.pt)
        check(f'{tag}: rebase sets are the hyperplanes that separate the anchors', tuple(sorted(obj.rebaseSet)) == separating(obj, obj.pt, other.pt) == tuple(sorted(other.rebaseSet)))
        ok3, ok4 = True, True
        for _ in range(60):
            x = rng.uniform(-1, 1, size=(d, 1))
            if np.min(np.abs(A @ x - b)) < 1e-6: continue
            lo, lb = label(obj, x), label(other, x)
            ok3 = ok3 and lo == separating(obj, x, obj.pt) and lb == separating(other, x, other.pt)
            ok4 = ok4 and obj.rebaseRegion(lb)[1] == lo and other.rebaseRegion(lo)[1] == lb
        check(f'{tag}: a label is the set of hyperplanes that separate the point from the anchor', ok3)
        check(f'{tag}: rebaseRegion translates labels between objects with different anchors', ok4)
        obj.clearRebase(); other.clearRebase()

# The situation that broke the TLL compiler: on the cube [0,1]^5, x0 = x4 passes through the centre
n = 5
fA = np.vstack([np.eye(n), -np.eye(n)]); fb = np.hstack([np.zeros(n), -np.ones(n)]).reshape(-1, 1)
centre = region_helpers.findInteriorPoint(np.hstack([-fb, fA]))
k = np.array([1.0, 0.0, 0.0, 0.0, -1.0])
own = region_helpers.flipConstraintsReducedMin(np.ones((0, n)), np.ones((0, 1)), centre.copy(), fA=fA, fb=fb)
own.insertHyperplane(k.copy(), np.array([0.0]))                     # as TLLnetIR.fromReLU does
merged = region_helpers.flipConstraintsReducedMin(np.array([[0.0, -1.0, 0.0, -1.0, 1.0]]), np.zeros((1, 1)), centre.copy(), fA=fA, fb=fb)
merged.insertHyperplane(-own.constraints[0, 1:].copy(), np.array([own.constraints[0, 0]]))     # as posetFastCharm.insertHyperplane does
check('cube: the centre of the cube is the anchor', np.allclose(centre.flatten(), 0.5))
own.setRebase(merged.pt)
rng = np.random.default_rng(0)
ok = True
for _ in range(200):
    x = rng.uniform(0, 1, size=(n, 1))
    if abs(float(k @ x.flatten())) < 1e-6: continue
    bit = tuple(i - 1 for i in label(merged, x) if i >= 1)           # the bit of the inserted hyperplane in the merged object
    ok = ok and own.rebaseRegion(bit)[1] == label(own, x)
check('cube: the bit of an inserted hyperplane is translated correctly to the hyperplane\'s own object', ok)

print(f'{"ALL TESTS PASSED" if len(failures) == 0 else f"{len(failures)} TEST(S) FAILED"}')
sys.exit(0 if len(failures) == 0 else 1)
