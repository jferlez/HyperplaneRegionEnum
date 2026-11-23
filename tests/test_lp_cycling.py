"""
Test (no charm4py) of the solvers and the fallback path of encapsulateLP.runLP, on an LP on which GLPK's simplex
cycles forever with the settings of this package (dual-then-primal simplex, scaling on, bound tolerances 1e-10).

The LP (tests/lp_cycling_case.npz: minimize obj.x subject to A x <= b, 19 constraints, 5 variables, two identical
rows) is an interior point search (region_helpers.findInteriorPointFull) that came up while the TLL compiler
reconciled the min terms of a sup-norm property on a network with small integer weights. Its optimal value is
-0.0600526 (so the region has an interior point at distance 0.06 from its faces). Without an iteration limit for
GLPK this call never returned; with it, GLPK gives up and the LP goes to the fallback solver.

Checked:
  * 'glpk' without a 'fallback' entry: solved by the default fallback (HiGHS), one fallback solve;
  * 'glpk' with 'fallback':None: no fallback, status 'unk';
  * 'glpk' with an explicit fallback chain (GLPK without scaling): solved by the first fallback;
  * 'highs' as the solver: solved directly, no fallback;
  * 'highs' that cannot finish (maxiter 0): solved by its default fallback GLPK (on an LP that GLPK solves), and
    not at all on the cycling LP, where both solvers are tried exactly once (no ping-pong between the defaults);
  * a fallback given as a solver name instead of a dictionary;
  * infeasible and unbounded LPs: the same conclusive status from both solvers, without a fallback solve;
  * an unknown solver name raises a ValueError;
  * an ordinary LP is solved without a fallback (interior point of the unit cube), with both solvers.

Usage: python3 tests/test_lp_cycling.py
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import encapsulateLP
import region_helpers

failures = []
def check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}{("   " + detail) if detail and not cond else ""}', flush=True)
    if not cond:
        failures.append(name)

case = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lp_cycling_case.npz'))
obj, A, b = case['obj'], case['A'], case['b']
OPT = -0.06005259929

def solve(lpopts, obj=obj, A=A, b=b):
    lp = encapsulateLP.encapsulateLP()
    t = time.time()
    status, x = lp.runLP(obj, A, b, lpopts=lpopts)
    return status, x, lp, time.time() - t

def isOptimum(status, x):
    return status == 'optimal' and x is not None and abs(float(obj @ x.flatten()) - OPT) < 1e-8 and bool(np.all(A @ x.flatten() <= b + 1e-9))

status, x, lp, elapsed = solve({'solver': 'glpk'})
check('cycling LP, glpk (default fallback HiGHS): optimal', isOptimum(status, x), f'status {status}')
check('cycling LP, glpk (default fallback HiGHS): one fallback solve, two LPs counted', lp.lpFallbacks == 1 and lp.lpCount == 2, f'{lp.lpFallbacks} fallbacks, {lp.lpCount} LPs')
check('cycling LP, glpk (default fallback HiGHS): returns quickly', elapsed < 10.0, f'{elapsed:.2f}s')

status, x, lp, _ = solve({'solver': 'glpk', 'fallback': None})
check('cycling LP, glpk with fallback None: unk, no fallback solve', status == 'unk' and x is None and lp.lpFallbacks == 0, f'status {status}, {lp.lpFallbacks} fallbacks')

status, x, lp, _ = solve({'solver': 'glpk', 'fallback': {'solver': 'glpk', 'scale': False, 'fallback': {'solver': 'highs'}}})
check('cycling LP, glpk with explicit fallback chain (glpk without scaling, then highs): optimal', isOptimum(status, x) and 1 <= lp.lpFallbacks <= 2, f'status {status}, {lp.lpFallbacks} fallbacks')

status, x, lp, _ = solve({'solver': 'glpk', 'fallback': 'highs'})
check('cycling LP, fallback given as the string "highs": optimal', isOptimum(status, x) and lp.lpFallbacks == 1, f'status {status}, {lp.lpFallbacks} fallbacks')

status, x, lp, _ = solve({'solver': 'highs'})
check('cycling LP, highs: optimal without a fallback solve', isOptimum(status, x) and lp.lpFallbacks == 0, f'status {status}, {lp.lpFallbacks} fallbacks')

status, x, lp, _ = solve({'solver': 'highs', 'maxiter': 0})
check('cycling LP, highs stopped at once (maxiter 0): default fallback glpk also fails; each solver tried once', status == 'unk' and x is None and lp.lpFallbacks == 1 and lp.lpCount == 2, f'status {status}, {lp.lpFallbacks} fallbacks, {lp.lpCount} LPs')

# An LP that GLPK solves: the interior point LP of the unit cube in R^4, written out (max t s.t. H [x; t] <= h)
n = 4
cubeA = np.vstack([np.hstack([-np.eye(n), np.ones((n, 1))]), np.hstack([np.eye(n), np.ones((n, 1))])])
cubeb = np.hstack([np.zeros(n), np.ones(n)])
cubeObj = np.zeros(n + 1); cubeObj[-1] = -1.0
for lpopts, nFall in (({'solver': 'glpk'}, 0), ({'solver': 'highs'}, 0), ({'solver': 'highs', 'maxiter': 0}, 1)):
    status, x, lp, _ = solve(lpopts, cubeObj, cubeA, cubeb)
    check(f'cube LP, {lpopts}: optimal with {nFall} fallback solve(s)', status == 'optimal' and abs(float(x.flatten()[-1]) - 0.5) < 1e-8 and lp.lpFallbacks == nFall, f'status {status}, {lp.lpFallbacks} fallbacks')

# Conclusive answers other than 'optimal' do not go to the fallback solver
infA = np.array([[1.0], [-1.0]]); infb = np.array([0.0, -1.0])           # x <= 0 and x >= 1
unbA = np.array([[-1.0]]); unbb = np.array([0.0])                        # min -x s.t. x >= 0
for solver in ('glpk', 'highs'):
    status, x, lp, _ = solve({'solver': solver}, np.array([1.0]), infA, infb)
    check(f'infeasible LP, {solver}: primal infeasible, no fallback solve', status == 'primal infeasible' and lp.lpFallbacks == 0, f'status {status}, {lp.lpFallbacks} fallbacks')
    status, x, lp, _ = solve({'solver': solver}, np.array([-1.0]), unbA, unbb)
    check(f'unbounded LP, {solver}: dual infeasible, no fallback solve', status == 'dual infeasible' and lp.lpFallbacks == 0, f'status {status}, {lp.lpFallbacks} fallbacks')

try:
    solve({'solver': 'nosuchsolver'})
    check('unknown solver raises ValueError', False)
except ValueError as e:
    check('unknown solver raises ValueError', 'unknown LP solver' in str(e), str(e))

# An ordinary LP through region_helpers (no fallback): the interior point of the unit cube
H = np.vstack([np.hstack([np.zeros((n, 1)), np.eye(n)]), np.hstack([np.ones((n, 1)), -np.eye(n)])])
for solver in ('glpk', 'highs'):
    lp2 = encapsulateLP.encapsulateLP()
    pt = region_helpers.findInteriorPoint(H, solver=solver, lpObj=lp2)
    check(f'interior point of the cube through region_helpers, {solver}: centre, no fallback solve', pt is not None and np.allclose(pt.flatten(), 0.5) and lp2.lpFallbacks == 0, f'pt {None if pt is None else pt.flatten()}, {lp2.lpFallbacks} fallbacks')

print(f'\n{"ALL TESTS PASSED" if len(failures) == 0 else str(len(failures)) + " TEST(S) FAILED"}')
sys.exit(0 if len(failures) == 0 else 1)
