"""
Randomized consistency test (no charm4py) of flipConstraints.insertHyperplane / removeHyperplanes when the
arrangement contains duplicate hyperplanes, duplicates of the fixed constraints and hyperplanes that do not
intersect the fixed polytope.

After every operation the test checks
  * the bookkeeping: hyperSet rows aligned with allConstraints[:allN], lengths of redundantFlips/flipMapN/nA/nb,
    nonRedundantHyperplanes/N/constraints, byte lengths;
  * that the non-redundant hyperplanes are exactly one representative of every distinct hyperplane that cuts the
    fixed polytope (and that each is a "unique" row of hyperSet);
  * expandRegion/collapseRegion on the regions of random sample points;
  * after a removal: that applyLastRemoval maps the (pre-removal) region of every sample point to its region in
    the new arrangement.

Usage: python3 tests/test_flipConstraints_removal.py [numSeeds] [--cls flipConstraints|flipConstraintsReducedMin] [--cover] [--quiet]
       (needs the same PYTHONPATH as the rest of the package, i.e. vectorSet has to be importable;
        --cover prints how often each kind of operation was exercised)
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import region_helpers


COVER = {}
def cover(ky, n=1): COVER[ky] = COVER.get(ky, 0) + n

def sameHyper(u, v):
    u = u / np.linalg.norm(u[1:]); v = v / np.linalg.norm(v[1:])
    return np.allclose(u, v, atol=1e-7) or np.allclose(u, -v, atol=1e-7)

class Model:
    """ Reference: the list of hyperplanes (rows [c0, a], meaning the zero set of c0 + a.x) in allN order, each
        with a kind: 'cut' (cuts the box), 'out' (misses the box) or 'fixed' (coincides with a box face) """
    def __init__(self, d, box):
        self.d, self.box = d, box
        self.rows, self.kinds = [], []
    def cls(self, i):
        return [j for j in range(len(self.rows)) if sameHyper(self.rows[i], self.rows[j])]
    def active(self):
        # one (unspecified) representative per class of 'cut' hyperplanes
        seen, groups = set(), []
        for i in range(len(self.rows)):
            if i in seen or self.kinds[i] != 'cut': continue
            g = self.cls(i); seen |= set(g); groups.append(g)
        return groups

def sideVals(H, x):
    return H[:, 0] + H[:, 1:] @ x.flatten()

def check(obj, model, samples, what, reducedMin):
    errs = []
    n = len(model.rows)
    nFixed = obj.allConstraints.shape[0] - obj.allN
    lens = {'allN': obj.allN, 'hyperSet.N': obj.hyperSet.N, 'redundantFlips': len(obj.redundantFlips), 'flipMapN': len(obj.flipMapN),
            'nA': obj.nA.shape[0], 'nb': obj.nb.shape[0], 'allNrms': obj.allNrms.shape[0] - nFixed}
    if any(v != n for v in lens.values()):
        return [f'[{what}] inconsistent sizes {lens}; expected {n}']
    if nFixed != 2 * model.d: errs.append(f'{nFixed} fixed constraints, expected {2 * model.d}')
    hsRows = obj.hyperSet.getRows() if n > 0 else np.zeros((0, model.d + 1))
    for i in range(n):
        if not sameHyper(obj.allConstraints[i], model.rows[i]): errs.append(f'allConstraints row {i} is not hyperplane {i}'); break
        if not sameHyper(hsRows[i], model.rows[i]): errs.append(f'hyperSet row {i} is not hyperplane {i}'); break
        if not sameHyper(np.hstack([-obj.nb[i], obj.nA[i]]), model.rows[i]): errs.append(f'nA/nb row {i} is not hyperplane {i}'); break
    if np.any(sideVals(obj.allConstraints, obj.pt) <= 0): errs.append('pt is not on the positive side of every row of allConstraints')
    nonRed = [int(i) for i in obj.nonRedundantHyperplanes]
    if nonRed != [int(i) for i in np.nonzero(obj.redundantFlips > 0)[0]]: errs.append('nonRedundantHyperplanes != nonzero(redundantFlips > 0)')
    if obj.N != len(nonRed): errs.append(f'N = {obj.N} but {len(nonRed)} non-redundant hyperplanes')
    expC = np.vstack([obj.allConstraints[nonRed, :], obj.allConstraints[obj.allN:, :]])
    if obj.constraints.shape != expC.shape or not np.array_equal(obj.constraints, expC): errs.append('constraints != allConstraints[nonRedundantHyperplanes] + fixed')
    if (obj.wholeBytes, obj.tailBits) != (obj.N // 8, obj.N % 8): errs.append('wholeBytes/tailBits do not match N')
    if (obj.wholeBytesAllN, obj.tailBitsAllN) != (obj.allN // 8, obj.allN % 8): errs.append('wholeBytesAllN/tailBitsAllN do not match allN')
    if not set(nonRed) <= set(int(i) for i in obj.hyperSet.uniqRowIdx): errs.append(f'non-redundant {nonRed} not all unique rows of hyperSet {list(obj.hyperSet.uniqRowIdx)}')
    # The non-redundant hyperplanes: one per class of hyperplanes that cut the box. (The base class does not
    # test whether a hyperplane cuts the fixed polytope, so there 'out' hyperplanes count as well.)
    wanted = model.active() if reducedMin else [g for g in
                 [model.cls(i) for i in range(n) if model.kinds[i] != 'fixed' and min(model.cls(i)) == i]]
    for g in wanted:
        if len(set(g) & set(nonRed)) != 1: errs.append(f'hyperplane class {g} has {len(set(g) & set(nonRed))} non-redundant representatives (non-redundant: {nonRed})')
    if len(nonRed) != len(wanted): errs.append(f'{len(nonRed)} non-redundant hyperplanes, expected {len(wanted)}')
    if not errs:
        for x in samples:
            rN = tuple(int(i) for i in np.nonzero(sideVals(obj.constraints[:obj.N], x) < 0)[0])
            rAll = tuple(int(i) for i in np.nonzero(sideVals(obj.allConstraints[:obj.allN], x) < 0)[0])
            # hyperplanes that are redundant for a reason other than being a duplicate are never flipped
            if not reducedMin: rAll = tuple(i for i in rAll if len(set(model.cls(i)) & set(nonRed)) > 0)
            e = tuple(int(i) for i in obj.expandRegion(rN))
            c = tuple(int(i) for i in obj.collapseRegion(rAll))
            if e != rAll: errs.append(f'expandRegion({rN}) = {e}, expected {rAll}'); break
            if c != rN: errs.append(f'collapseRegion({rAll}) = {c}, expected {rN}'); break
    return [f'[{what}] {e}' for e in errs]


def runSeed(seed, clsName, nOps=25, verbose=False):
    rng = np.random.default_rng(seed)
    reducedMin = (clsName == 'flipConstraintsReducedMin')
    d = int(rng.integers(2, 4))
    box = 2.0
    fA = np.vstack([np.eye(d), -np.eye(d)]); fb = -box * np.ones((2 * d, 1))
    model = Model(d, box)

    def newHyper():
        kind = rng.choice(['cut', 'cut', 'cut', 'dup', 'dup', 'out', 'fixed'])
        if kind == 'dup' and len(model.rows) > 0:
            i = int(rng.integers(len(model.rows)))
            return model.rows[i] * float(rng.choice([-1.0, 1.0, 2.0, -0.5])), model.kinds[i]
        a = rng.normal(size=d)
        if abs(a[-1]) < 0.2: a[-1] = 0.5        # (a zero last coefficient is a separate limitation of vectorSet)
        if kind == 'fixed':
            j = int(rng.integers(d)); a = np.zeros(d); a[j] = float(rng.choice([-1.0, 1.0]))
            return np.hstack([[box], a]) * float(rng.choice([1.0, -2.0])), 'fixed'
        if kind == 'out':
            return np.hstack([[-(np.sum(np.abs(a)) * box + rng.uniform(0.5, 2.0))], a]) * float(rng.choice([-1.0, 1.0])), 'out'
        x0 = rng.uniform(-0.8 * box, 0.8 * box, size=d)
        return np.hstack([[-a @ x0], a]), 'cut'

    init = [newHyper() for _ in range(int(rng.integers(1, 6)))]
    for r, k in init: model.rows.append(r); model.kinds.append(k)
    # base point: inside the box and not on any hyperplane
    while True:
        pt = rng.uniform(-0.9 * box, 0.9 * box, size=(d, 1))
        if all(abs(r[0] + r[1:] @ pt.flatten()) > 1e-3 for r, k in init): break
    nA = np.array([r[1:] for r, k in init]); nb = np.array([[-r[0]] for r, k in init])
    obj = getattr(region_helpers, clsName)(nA, nb, pt, fA=fA, fb=fb, tol=1e-9, rTol=1e-9)

    def drawSamples():
        out = []
        while len(out) < 40:
            x = rng.uniform(-box, box, size=d)
            if all(abs(r[0] + r[1:] @ x) > 1e-4 * np.linalg.norm(r[1:]) for r in model.rows): out.append(x)
        return out

    log = [f'{clsName}: init with {[k for r, k in init]} (d={d})']
    errs = check(obj, model, drawSamples(), log[-1], reducedMin)
    for op in range(nOps):
        if errs: break
        kind = rng.choice(['insert', 'insert', 'remove', 'copy'])
        try:
            if kind == 'insert' or obj.N == 0:
                r, k = newHyper()
                isNew = not any(sameHyper(r, q) for q in model.rows)
                expRet = isNew and k != 'fixed'
                log.append(f'insertHyperplane({k}{"" if isNew else " (duplicate)"}) expect {expRet}')
                if abs(r[0] + r[1:] @ obj.pt.flatten()) < 1e-6: continue
                ret = obj.insertHyperplane(r[1:].copy(), np.float64(-r[0]))
                cover('insert: ' + k + ('' if isNew else ' duplicate') + (' (opposite orientation to pt)' if r[0] + r[1:] @ obj.pt.flatten() < 0 else ''))
                model.rows.append(r); model.kinds.append(k)
                if bool(ret) != expRet: errs.append(f'[{log[-1]}] returned {ret}')
            elif kind == 'copy':
                log.append('copy (serialize/deserialize)')
                obj = obj.copy()
            else:
                allN = bool(rng.random() < 0.3)
                inclDup = bool(rng.random() < 0.6)
                total = obj.allN if allN else obj.N
                hl = sorted(int(i) for i in rng.choice(total, size=int(rng.integers(1, min(3, total) + 1)), replace=False))
                idxAll = hl if allN else [int(obj.nonRedundantHyperplanes[i]) for i in hl]
                rem = sorted(set(j for i in idxAll for j in model.cls(i))) if inclDup else idxAll
                log.append(f'removeHyperplanes({hl}, allN={allN}, includeDup={inclDup}): rows {idxAll} -> removes {rem} of {len(model.rows)}; kinds {[model.kinds[i] for i in idxAll]}')
                nonRedSet = set(int(i) for i in obj.nonRedundantHyperplanes)
                cover(f'remove: allN={allN}, includeDup={inclDup}')
                if len(rem) > len(idxAll): cover('remove: duplicates removed along with the listed hyperplanes')
                if any(i in nonRedSet and len(set(model.cls(i)) - set(rem)) > 0 for i in rem): cover('remove: active hyperplane removed, a duplicate takes over')
                if any(model.kinds[i] != 'cut' for i in rem): cover('remove: includes fixed-duplicate/outside hyperplanes')
                if len(rem) == len(model.rows): cover('remove: all hyperplanes')
                samples = drawSamples()
                preN = [tuple(int(i) for i in np.nonzero(sideVals(obj.constraints[:obj.N], x) < 0)[0]) for x in samples]
                obj.removeHyperplanes(list(hl), allN=allN, includeDup=inclDup)
                model.rows = [r for i, r in enumerate(model.rows) if i not in rem]
                model.kinds = [k for i, k in enumerate(model.kinds) if i not in rem]
                errs += check(obj, model, samples, log[-1], reducedMin)
                if not errs:
                    for x, rN in zip(samples, preN):
                        expN = tuple(int(i) for i in np.nonzero(sideVals(obj.constraints[:obj.N], x) < 0)[0])
                        got = tuple(int(i) for i in obj.applyLastRemoval(rN))
                        if len(rN) > 0 and got != rN: cover('remove: region re-encoded (sample points)')
                        if got != expN: errs.append(f'[{log[-1]}] applyLastRemoval({rN}) = {got}, expected {expN}'); break
                        gotB = obj.applyLastRemoval(region_helpers.tupToBytes(rN, obj.lastRemoval.wholeBytes, obj.lastRemoval.tailBits))
                        if tuple(obj.bytesToList(gotB)) != expN: errs.append(f'[{log[-1]}] applyLastRemoval(bytes of {rN}) = {tuple(obj.bytesToList(gotB))}, expected {expN}'); break
                continue
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)[-1]
            errs.append(f'[{log[-1]}] raised {type(e).__name__} at {tb.name}:{tb.lineno}: {str(e)[:150]}')
            break
        errs += check(obj, model, drawSamples(), log[-1], reducedMin)
    if errs and verbose:
        print(f'seed {seed}:'); [print('   ', l) for l in log[-3:]]; [print('    ->', e.split("] ", 1)[-1]) for e in errs[:3]]
    return errs


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    nSeeds = int(args[0]) if len(args) > 0 else 300
    classes = [sys.argv[sys.argv.index('--cls') + 1]] if '--cls' in sys.argv else ['flipConstraintsReducedMin', 'flipConstraints']
    allOk = True
    for clsName in classes:
        failed = {}
        for seed in range(nSeeds):
            errs = runSeed(seed, clsName, verbose=(len(failed) < 3 and '--quiet' not in sys.argv))
            if errs: failed[seed] = errs
        kinds = {}
        for errs in failed.values():
            e = errs[0].split('] ', 1)[-1]
            ky = (e.split(':')[0] + ':' + e.split(':')[1] if e.startswith('raised') else e.split('(')[0].split(' {')[0])[:70]
            kinds[ky] = kinds.get(ky, 0) + 1
        print(f'{clsName}: {nSeeds - len(failed)} of {nSeeds} random operation sequences passed')
        for ky, cnt in sorted(kinds.items(), key=lambda kv: -kv[1]): print(f'   {cnt:4d} x first failure: {ky}')
        allOk = allOk and not failed
    if '--cover' in sys.argv:
        for ky in sorted(COVER): print(f'   coverage {COVER[ky]:6d} x {ky}')
    print(f'######## {"ALL TESTS PASSED" if allOk else "SOME TESTS FAILED"} ########')
    sys.exit(0 if allOk else 1)
