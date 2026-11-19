"""
Checks that flipConstraints objects pickled by older versions of region_helpers/vectorSet still work when they
are unpickled with the current code (flipConstraints.__setstate__ / vectorSet.__setstate__).

tests/legacy_flipConstraints.pkl was written by region_helpers.py at commit b23efe5 with vectorSet.py at commit
5b502d0: flipConstraintsReducedMin objects built with the constructor and insertHyperplane, with the hyperplanes
that were put into them. About half of the hyperplanes have a zero last coefficient, and many are duplicates, so
the fixture contains
  * objects whose redundantFlips is shorter than allN (a duplicate was inserted last), and
  * objects in which the old code did not recognise a duplicate (zero last coefficient, opposite orientation)
    and counted the same hyperplane twice. Those cannot be repaired when they are loaded: the regions of
    existing tables are encoded with that hyperplane count. They must load with a warning and unchanged N.
All other objects must pass the same checks as objects built by the current code, including removing
hyperplanes and inserting further duplicates.

Usage: python3 tests/test_legacy_upgrade.py [fixture.pkl]
"""
import sys, os, pickle, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_helpers
import test_flipConstraints_removal as T

fixture = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'legacy_flipConstraints.pkl')
objs = pickle.load(open(fixture, 'rb'))
rng = np.random.default_rng(0)
problems = []
stats = {'objects': 0, 'with a short redundantFlips': 0, 'with a duplicate counted twice (warning expected)': 0, 'removals checked': 0}
for k, rec in enumerate(objs):
    errs = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        obj = pickle.loads(rec['pickle'])
    obj.deserialize()
    model = T.Model(rec['d'], rec['box'])
    model.rows = [np.array(r) for r in rec['rows']]; model.kinds = ['cut' for r in rec['rows']]
    stats['objects'] += 1
    stats['with a short redundantFlips'] += int(rec['staleRedundantFlips'] > 0)
    def drawSamples():
        out = []
        while len(out) < 40:
            x = rng.uniform(-rec['box'], rec['box'], size=rec['d'])
            if all(abs(r[0] + r[1:] @ x) > 1e-4 * np.linalg.norm(r[1:]) for r in model.rows): out.append(x)
        return out
    try:
        if len(obj.redundantFlips) != obj.allN: errs.append(f'redundantFlips has {len(obj.redundantFlips)} entries for allN = {obj.allN}')
        if obj.N != rec['N']: errs.append(f'N changed from {rec["N"]} to {obj.N}')
        if rec['duplicateCountedTwice']:
            stats['with a duplicate counted twice (warning expected)'] += 1
            if len(w) == 0: errs.append('no warning for an object in which a duplicate hyperplane is counted twice')
        else:
            if len(w) > 0: errs.append(f'unexpected warning: {str(w[0].message)[:100]}')
            errs += T.check(obj, model, drawSamples(), 'after load', True)
            if not errs and obj.N > 0:
                # the way applyReLU removes masked hyperplanes
                hl = sorted(int(i) for i in rng.choice(obj.N, size=int(rng.integers(1, min(3, obj.N) + 1)), replace=False))
                rem = sorted(set(j for i in hl for j in model.cls(int(obj.nonRedundantHyperplanes[i]))))
                samples = drawSamples()
                preN = [tuple(int(i) for i in np.nonzero(T.sideVals(obj.constraints[:obj.N], x) < 0)[0]) for x in samples]
                obj.removeHyperplanes(list(hl), allN=False, includeDup=True)
                model.rows = [r for i, r in enumerate(model.rows) if i not in rem]; model.kinds = model.kinds[:len(model.rows)]
                errs += T.check(obj, model, samples, f'after removing {hl}', True)
                for x, rN in zip(samples, preN):
                    expN = tuple(int(i) for i in np.nonzero(T.sideVals(obj.constraints[:obj.N], x) < 0)[0])
                    if tuple(int(i) for i in obj.applyLastRemoval(rN)) != expN: errs.append(f'applyLastRemoval({rN}) != {expN}'); break
                stats['removals checked'] += 1
            if not errs and len(model.rows) > 0:
                r = -2.0 * model.rows[int(rng.integers(len(model.rows)))]
                if obj.insertHyperplane(r[1:].copy(), np.float64(-r[0])): errs.append('a duplicate (negated) hyperplane was inserted as a new one')
                model.rows.append(r); model.kinds.append('cut')
                errs += T.check(obj, model, drawSamples(), 'after inserting a duplicate', True)
    except Exception as e:
        errs.append(f'raised {type(e).__name__}: {str(e)[:150]}')
    if errs: problems.append((k, errs))
for k, errs in problems[:5]: print(f'object {k}: {errs[:3]}')
print(f'{len(objs) - len(problems)} of {len(objs)} legacy objects behave correctly after loading')
print('   ' + ', '.join(f'{v} {ky}' for ky, v in stats.items()))
print(f'######## {"ALL TESTS PASSED" if not problems else "SOME TESTS FAILED"} ########')
sys.exit(0 if not problems else 1)
