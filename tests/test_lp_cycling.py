"""
Test (no charm4py) that encapsulateLP.runLP returns, with the right answer, on an LP on which GLPK's simplex cycles
forever with the settings of this package (dual-then-primal simplex, scaling on, bound tolerances 1e-10).

The LP (tests/lp_cycling_case.npz: minimize obj.x subject to A x <= b, 19 constraints, 5 variables, two identical
rows) is an interior point search (region_helpers.findInteriorPointFull) that came up while the TLL compiler
reconciled the min terms of a sup-norm property on a network with small integer weights. Its optimal value is
-0.0600526 (so the region has an interior point at distance 0.06 from its faces). Before the iteration limit and
the retries in encapsulateLP.runLP, this call never returned.

Usage: python3 tests/test_lp_cycling.py
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import encapsulateLP

case = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lp_cycling_case.npz'))
obj, A, b = case['obj'], case['A'], case['b']
lp = encapsulateLP.encapsulateLP()
t = time.time()
status, x = lp.runLP(obj, A, b, lpopts={'solver': 'glpk'})
elapsed = time.time() - t
ok = status == 'optimal' and x is not None and abs(float(obj @ x.flatten()) + 0.06005259929) < 1e-8 and bool(np.all(A @ x.flatten() <= b + 1e-9))
print(f'status {status}; objective {None if x is None else float(obj @ x.flatten())}; {elapsed:.2f}s; retries {getattr(lp, "lpRetries", 0)}')

# an ordinary LP is unaffected (no retries): the interior point of the unit cube
n = 4
H = np.vstack([np.hstack([np.zeros((n, 1)), np.eye(n)]), np.hstack([np.ones((n, 1)), -np.eye(n)])])
import region_helpers
lp2 = encapsulateLP.encapsulateLP()
pt = region_helpers.findInteriorPoint(H, lpObj=lp2)
ok2 = pt is not None and np.allclose(pt.flatten(), 0.5) and getattr(lp2, 'lpRetries', 0) == 0
print(f'interior point of the cube: {None if pt is None else pt.flatten()}; retries {getattr(lp2, "lpRetries", 0)}')
print('ALL TESTS PASSED' if ok and ok2 else 'TEST FAILED')
sys.exit(0 if ok and ok2 else 1)
