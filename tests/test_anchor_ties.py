"""
Test (no charm4py) of what flipConstraints guarantees when a hyperplane passes exactly through the anchor point,
and of the translation of region labels between objects with different anchor points (setRebase/rebaseRegion).

A region is labeled by the hyperplanes that separate it from the anchor point of the object. Code that carries a
label from one object to another (e.g. the table merges of the TLL compiler) used to assume that both objects have
their anchor in the same region, which is true as long as they were all created with the same anchor point and no
hyperplane passes through it. For a cube and hyperplanes like x0 = x4 that assumption fails: the anchor (the centre
of the cube) lies on the hyperplane, each object picks a side, and moves its own anchor there.

Checked here, for random arrangements and for arrangements with hyperplanes through the anchor:
  1. tie rule (flipConstraints._tieSigns): a hyperplane through the anchor is oriented such that the first
     non-zero entry of its normal is positive, i.e. towards the point  anchor + e*(1, e, e^2, ...), both in the
     constructor and in insertHyperplane. The root region is therefore the same cell however the hyperplanes are
     written (rows negated, scaled, reordered, duplicated);
  2. whatever the number of hyperplanes through the anchor, construction/insertion succeeds, and afterwards the
     anchor is strictly inside the root region (the region with the empty label). (With several hyperplanes
     through one point, the sides "as given" need not intersect, e.g. three lines in the plane.)
  3. the label of a point, computed from the signs of the constraints, is the set of hyperplanes that separate
     the point from the anchor;
  4. for two objects A and B with the same hyperplanes but different anchors (and therefore different
     orientations and labels), A.setRebase(B.pt) followed by A.rebaseRegion(label in B) is the label in A.

  5. special hyperplanes through the anchor: normals with leading zeros (axis-aligned), a leading entry that is
     rounding noise, duplicates (also with the opposite orientation or another scale), 20 hyperplanes through
     one point, a single input, zero normals (without normalization), and an anchor on the boundary of the fixed
     constraints (on a face, with a hyperplane that coincides with that face, and at a vertex).

Usage: python3 tests/test_anchor_ties.py [numSeeds]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import region_helpers

failures = []
numChecks = 0
def check(name, cond):
    global numChecks
    numChecks += 1
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

def lexPositive(row):
    # True if the first non-zero entry of the normal of the stored row [c0, a] is positive
    a = row[1:]
    nz = np.nonzero(np.abs(a) > 1e-9 * np.max(np.abs(a)))[0]
    return len(nz) > 0 and a[nz[0]] > 0

def anchorInside(obj):
    v = obj.constraints[:obj.N, 0] + obj.constraints[:obj.N, 1:] @ obj.pt.flatten()
    vf = obj.constraints[obj.N:, 0] + obj.constraints[obj.N:, 1:] @ obj.pt.flatten()
    return (obj.N == 0 or float(np.min(v)) > obj.tol) and (len(vf) == 0 or float(np.min(vf)) > obj.tol)

def sameRootRegion(objA, objB):
    # The anchor of each object is strictly inside the root region of the other
    def inside(obj, x):
        v = obj.constraints[:obj.N, 0] + obj.constraints[:obj.N, 1:] @ x.flatten()
        return obj.N == 0 or float(np.min(v)) > 0
    return inside(objA, objB.pt) and inside(objB, objA.pt)

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
        check(f'{tag}: constructor orients every hyperplane through the anchor by the first non-zero entry of its normal', all(lexPositive(allRows[i]) for i in range(nGen, nGen + nTie)))
        check(f'{tag}: the anchor is strictly inside the root region', anchorInside(obj))
        # the same arrangement written differently: rows negated, scaled, reordered, and some of them given twice
        perm = rng.permutation(nGen + nTie)
        scl = rng.choice([-1.0, 1.0], size=(nGen + nTie, 1)) * rng.uniform(0.2, 5.0, size=(nGen + nTie, 1))
        A2 = np.vstack([(scl * A)[perm], -A[nGen:nGen + 1]]); b2 = np.vstack([(scl * b)[perm], -b[nGen:nGen + 1]])
        obj2 = cls(A2.copy(), b2.copy(), centre.copy(), fA=fA, fb=fb)
        check(f'{tag}: the root region does not depend on how the hyperplanes are written', obj2.N == obj.N and anchorInside(obj2) and sameRootRegion(obj, obj2))
        objN = cls(A.copy(), b.copy(), centre.copy(), fA=fA, fb=fb, normalize=1.0)
        check(f'{tag}: ... nor on normalization', anchorInside(objN) and sameRootRegion(obj, objN))
        check(f'{tag}: the anchor was moved off the hyperplanes', not np.array_equal(obj.pt, centre))

        # ---- the same hyperplanes inserted one at a time into an object without hyperplanes ----
        ins = cls(np.ones((0, d)), np.ones((0, 1)), centre.copy(), fA=fA, fb=fb)
        for i in range(nGen + nTie):
            ins.insertHyperplane(A[i].copy(), b[i].copy())
        insRows = ins.allConstraints[:ins.allN]
        # (only the first of them is a tie: it moves the anchor off the centre)
        check(f'{tag}: the anchor is strictly inside the root region after insertions', anchorInside(ins))
        # a hyperplane inserted through the anchor (the first one into an empty object: its anchor is the centre)
        for sgn in (1.0, -1.0):
            one = cls(np.ones((0, d)), np.ones((0, 1)), centre.copy(), fA=fA, fb=fb)
            one.insertHyperplane(sgn * A[nGen].copy(), sgn * b[nGen].copy())
            check(f'{tag}: insertHyperplane orients a hyperplane through the anchor by the first non-zero entry of its normal (given with sign {sgn:+.0f})', lexPositive(one.allConstraints[0]) and anchorInside(one))

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

# ---- special hyperplanes through the anchor ----
def build(rows, pt, fA, fb, **kw):
    rows = np.array(rows, dtype=np.float64)
    return region_helpers.flipConstraints(rows[:, 1:].copy(), -rows[:, :1].copy(), np.array(pt, dtype=np.float64).reshape(-1, 1), fA=fA, fb=fb, **kw)

def cube(d, lo=-1.0, hi=1.0):
    return np.vstack([np.eye(d), -np.eye(d)]), np.hstack([lo * np.ones(d), -hi * np.ones(d)]).reshape(-1, 1)

for kw in ({}, {'normalize': 1.0}):
    tag = f'special ({"normalize=1.0" if kw else "no normalization"})'
    fA3, fb3 = cube(3)
    # (rows are [c, a]:  c + a.x >= 0)
    o = build([[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, -2, 0], [0, 0, 1, 1], [0, -1, 0, 0]], [0, 0, 0], fA3, fb3, **kw)
    check(f'{tag}: axis-aligned normals with leading zeros', o.N == 4 and anchorInside(o) and all(lexPositive(r) for r in o.allConstraints[:o.allN]))
    o = build([[0, 1e-17, 1, -1], [0, -1e-17, -1, 1], [0, 0, 3, -3]], [0, 0, 0], fA3, fb3, **kw)
    check(f'{tag}: a leading entry that is rounding noise does not orient the hyperplane; the three copies are one hyperplane', o.N == 1 and anchorInside(o) and o.constraints[0, 2] > 0)
    o = build([[0, 1, 2, 3], [0, -1, -2, -3], [0, 0.5, 1, 1.5], [0, -4, -8, -12]], [0, 0, 0], fA3, fb3, **kw)
    check(f'{tag}: duplicates with opposite orientations and scales', o.N == 1 and anchorInside(o) and lexPositive(o.constraints[0]))
    rng = np.random.default_rng(1)
    for d in (2, 3, 5):
        fAd, fbd = cube(d)
        o = build(np.hstack([np.zeros((20, 1)), rng.normal(size=(20, d))]), np.zeros(d), fAd, fbd, **kw)
        check(f'{tag}: 20 hyperplanes through the anchor, d = {d}', o.N == 20 and anchorInside(o))
        # ... through an anchor that is not the origin, and with integer normals (many parallel faces of cells)
        c = rng.uniform(-0.5, 0.5, size=(d,))
        Aint = rng.integers(-2, 3, size=(20, d)).astype(np.float64); Aint = Aint[np.any(Aint != 0, axis=1)]
        o = build(np.hstack([-(Aint @ c).reshape(-1, 1), Aint]), c, fAd, fbd, **kw)
        check(f'{tag}: integer hyperplanes through an anchor off the origin, d = {d}', anchorInside(o) and all(lexPositive(r) for r in o.constraints[:o.N]))
    fA1, fb1 = cube(1)
    o = build([[0, 1], [0, -1], [0, 2.5], [-0.5, 1]], [0], fA1, fb1, **kw)
    check(f'{tag}: a single input, the same point given three times', o.N == 2 and anchorInside(o) and o.pt[0, 0] > 0)
    # anchor on a face of the fixed constraints (x0 = 1): a hyperplane that coincides with the face, one that is
    # transversal to it, and one that only touches the cube there
    fA2, fb2 = cube(2)
    o = build([[-1, 1, 0], [0, 0, -1], [-1, 1, 0.0]], [1, 0], fA2, fb2, **kw)
    check(f'{tag}: anchor on a face; a hyperplane equal to the face is oriented into the cube (and is redundant)', o.N == 1 and anchorInside(o) and o.pt[1, 0] > 0)
    o = build([[1, -1, 0], [0, 0, 1]], [1, 0], fA2, fb2, **kw)
    check(f'{tag}: anchor on a face; the face given with the other orientation', o.N == 1 and anchorInside(o))
    # anchor at a vertex (1, 1): the diagonal through it, and a supporting line that touches only the vertex
    o = build([[0, -1, 1], [-2, 1, 1], [-1, 1, 0]], [1, 1], fA2, fb2, **kw)
    check(f'{tag}: anchor at a vertex of the cube', anchorInside(o) and o.pt[0, 0] > o.pt[1, 0])
    # the same through insertHyperplane
    o = region_helpers.flipConstraints(np.ones((0, 2)), np.ones((0, 1)), np.array([[1.0], [0.0]]), fA=fA2, fb=fb2, **kw)
    o.insertHyperplane(np.array([0.0, -1.0]), np.array([0.0]))
    check(f'{tag}: insertHyperplane through an anchor on a face', o.N == 1 and anchorInside(o) and o.pt[1, 0] > 0)

# zero normals (constant "constraints") are only accepted without normalization; they have no side to choose
fA2, fb2 = cube(2)
o = build([[0.3, 0, 0], [0, 1, -1], [0, -1, -1]], [0, 0], fA2, fb2)
v = o.constraints[:, 0] + o.constraints[:, 1:] @ o.pt.flatten()
check('special: a zero normal next to two hyperplanes through the anchor is left as given', np.array_equal(o.allConstraints[0], [0.3, 0, 0]) and float(np.min(v)) > o.tol)
check('special: _tieSigns leaves a zero normal as given and orients the others', region_helpers.flipConstraints._tieSigns(np.array([[0.0, 0.0], [0.0, -2.0], [3.0, -1.0]]), np.zeros((2, 1))).tolist() == [1, -1, 1])

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

print(f'{numChecks} checks: {"ALL TESTS PASSED" if len(failures) == 0 else f"{len(failures)} TEST(S) FAILED"}')
sys.exit(0 if len(failures) == 0 else 1)
