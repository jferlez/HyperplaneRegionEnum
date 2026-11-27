"""
Test (no charm4py) of the two ways in which the solver 'glpk' of encapsulateLP.runLP calls GLPK: the cvxopt
binding (the default if cvxopt has GLPK) and the scikit-glpk binding (see encapsulateLP.glpkBinding).

Checked:
  * the same status and the same optimal value from both bindings on interior point LPs
    (region_helpers.findInteriorPointFull) of random regions, feasible and infeasible, in several sizes;
  * infeasible and unbounded LPs: the same conclusive status from both, without a fallback solve;
  * the options are passed on: an iteration limit of 0 is inconclusive ('unk' without a fallback, solved by
    the default fallback otherwise) with both bindings, and every simplex method/pricing/ratio test is accepted
    without a warning;
  * the LP of tests/test_lp_cycling.py, on which GLPK cycles when the problem is scaled: solved at once by the
    cvxopt binding (which does not scale), by the fallback solver with the scikit-glpk binding;
  * which binding is used: the default, lpopts['glpkBinding'], the module default, and the cases in which the
    cvxopt binding cannot do what is asked ('scale', 'init_basis', 'basis_fac', 'exact', ratio test 'flip', an LP
    with a non-finite entry, cvxopt without GLPK), which all go to scikit-glpk;
  * an LP without constraint rows is answered by runLP itself, for every solver (GLPK would abort the process):
    optimal at x = 0 for a zero objective, unbounded otherwise;
  * limits that do not fit in a C int are clamped (GLPK would abort the process), and an unknown binding raises;
  * the time per LP of both bindings (reported, not checked).

Usage: python3 tests/test_glpk_binding.py
"""
import sys, os, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import encapsulateLP
import region_helpers

failures = []
def check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}{("   " + str(detail)) if detail != "" and not cond else ""}', flush=True)
    if not cond:
        failures.append(name)

if encapsulateLP.cvxoptGlpk is None:
    print('cvxopt has no GLPK here: the solver \'glpk\' can only use the scikit-glpk binding. Nothing to compare.')
    check('default binding without cvxopt.glpk is scikit', encapsulateLP.glpkBinding == 'scikit')
    sys.exit(0 if len(failures) == 0 else 1)

# Which binding a solve went through
calls = {'cvxopt': 0, 'scikit': 0}
origCvx, origScikit = encapsulateLP.cvxoptGlpk.lp, encapsulateLP.glpk
def countedCvx(*a, **k):
    calls['cvxopt'] += 1
    return origCvx(*a, **k)
def countedScikit(*a, **k):
    calls['scikit'] += 1
    return origScikit(*a, **k)
class _Proxy:
    # stands in for the module cvxopt.glpk, counting the calls of lp
    lp = staticmethod(countedCvx)
encapsulateLP.cvxoptGlpk = _Proxy
encapsulateLP.glpk = countedScikit

def solve(lpopts, obj, A, b):
    lp = encapsulateLP.encapsulateLP()
    before = dict(calls)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        status, x = lp.runLP(obj, A, b, lpopts=lpopts)
    used = {ky: calls[ky] - before[ky] for ky in calls}
    return status, x, lp, used, [str(w.message) for w in wl]

check('default binding is cvxopt', encapsulateLP.glpkBinding == 'cvxopt', encapsulateLP.glpkBinding)

# ---- 1. Interior point LPs of random regions: max t  s.t.  c_i + a_i . x >= t,  0 <= t <= 1
rng = np.random.default_rng(0)
def interiorLP(m, d, feasible=True):
    q = rng.uniform(-1, 1, size=d)
    Ahyp = rng.normal(size=(m, d)); Ahyp /= np.linalg.norm(Ahyp, axis=1, keepdims=True)
    c0 = -(Ahyp @ q) + np.abs(rng.normal(scale=0.5, size=m))
    H = np.vstack([np.hstack([c0.reshape(-1, 1), Ahyp]), np.hstack([2.0 * np.ones((2 * d, 1)), np.vstack([np.eye(d), -np.eye(d)])])])
    if not feasible:
        # two opposite half-spaces that do not meet
        H = np.vstack([H, np.hstack([[-1.0 - q[0]], np.eye(d)[0]]), np.hstack([[-1.0 + q[0]], -np.eye(d)[0]])])
    n = d; N = H.shape[0]
    obj = np.zeros(n + 1); obj[n] = -1
    cons = np.hstack([np.hstack([H[:, 0], 1.0, 0.0]).reshape(-1, 1), np.vstack([np.hstack([-H[:, 1:], np.ones((N, 1))]), -obj, obj])])
    cons = cons / np.linalg.norm(cons, axis=1).reshape(-1, 1)
    return cons[-1, 1:], cons[:, 1:], cons[:, 0]
nLP = 0; worst = 0.0; statusDiff = 0; wrongBinding = 0; infeasibleSeen = 0; infeasibleExpected = 0
for (m, d) in [(4, 1), (10, 2), (25, 3), (60, 3), (40, 6), (150, 5)]:
    for k in range(60):
        obj, A, b = interiorLP(m, d, feasible=(k % 4 != 3))
        infeasibleExpected += (k % 4 == 3)
        s1, x1, lp1, used1, _ = solve({'solver': 'glpk', 'fallback': None}, obj, A, b)
        s2, x2, lp2, used2, _ = solve({'solver': 'glpk', 'glpkBinding': 'scikit', 'fallback': None}, obj, A, b)
        nLP += 1
        wrongBinding += (used1 != {'cvxopt': 1, 'scikit': 0}) + (used2 != {'cvxopt': 0, 'scikit': 1})
        # (the LP of an empty region is infeasible, because t >= 0 is one of its constraints)
        if s1 != s2 or s1 not in ('optimal', 'primal infeasible'):
            statusDiff += 1
        elif s1 == 'primal infeasible':
            infeasibleSeen += 1
            if x1 is not None or x2 is not None: statusDiff += 1
        else:
            worst = max(worst, abs(float(obj @ x1.flatten()) - float(obj @ x2.flatten())))
            if np.max(A @ x1.flatten() - b) > 1e-8: statusDiff += 1
check(f'{nLP} interior point LPs: each solve used the binding that was asked for', wrongBinding == 0, wrongBinding)
check(f'{nLP} interior point LPs ({infeasibleSeen} of empty regions): the same status from both bindings, and the same optimal value', \
      statusDiff == 0 and worst < 1e-8 and infeasibleSeen == infeasibleExpected, f'{statusDiff} differ, max difference {worst:.2e}, {infeasibleSeen} of {infeasibleExpected} infeasible')

# ---- 2. Conclusive answers other than 'optimal'
infA = np.array([[1.0], [-1.0]]); infb = np.array([0.0, -1.0])           # x <= 0 and x >= 1
unbA = np.array([[-1.0]]); unbb = np.array([0.0])                        # min -x s.t. x >= 0
for binding in ('cvxopt', 'scikit'):
    status, x, lp, used, _ = solve({'solver': 'glpk', 'glpkBinding': binding}, np.array([1.0]), infA, infb)
    check(f'infeasible LP, {binding}: primal infeasible, no fallback solve', status == 'primal infeasible' and x is None and lp.lpFallbacks == 0 and used[binding] == 1, f'status {status}, {lp.lpFallbacks} fallbacks, {used}')
    status, x, lp, used, _ = solve({'solver': 'glpk', 'glpkBinding': binding}, np.array([-1.0]), unbA, unbb)
    check(f'unbounded LP, {binding}: dual infeasible, no fallback solve', status == 'dual infeasible' and x is None and lp.lpFallbacks == 0 and used[binding] == 1, f'status {status}, {lp.lpFallbacks} fallbacks, {used}')

# ---- 3. Options reach GLPK
obj, A, b = interiorLP(25, 3)
ref = float(obj @ solve({'solver': 'glpk'}, obj, A, b)[1].flatten())
for binding in ('cvxopt', 'scikit'):
    status, x, lp, used, _ = solve({'solver': 'glpk', 'glpkBinding': binding, 'maxit': 0, 'fallback': None}, obj, A, b)
    check(f'iteration limit 0, {binding}, no fallback: unk', status == 'unk' and x is None and lp.lpFallbacks == 0 and used[binding] == 1, f'status {status}, {used}')
    status, x, lp, used, _ = solve({'solver': 'glpk', 'glpkBinding': binding, 'maxit': 0}, obj, A, b)
    check(f'iteration limit 0, {binding}: solved by the default fallback (HiGHS)', status == 'optimal' and abs(float(obj @ x.flatten()) - ref) < 1e-8 and lp.lpFallbacks == 1 and lp.lpCount == 2, f'status {status}, {lp.lpFallbacks} fallbacks')
nOpt = 0; warned = []
for method in ('primal', 'dual', 'dualp'):
    for steep in (True, False):
        for ratio in ('relax', 'norelax'):
            for presolve in (False, True):
                status, x, lp, used, wl = solve({'solver': 'glpk', 'method': method, 'steep': steep, 'ratio': ratio, 'presolve': presolve, 'tol_piv': 1e-9, 'timeout': 60000, 'fallback': None}, obj, A, b)
                nOpt += (status == 'optimal' and abs(float(obj @ x.flatten()) - ref) < 1e-8 and used == {'cvxopt': 1, 'scikit': 0})
                warned += wl
check('every simplex method / pricing / ratio test / presolve setting: optimal through the cvxopt binding, no warning', nOpt == 24 and len(warned) == 0, f'{nOpt} of 24, warnings {warned[:2]}')
o = encapsulateLP.cvxoptGlpkOptions({'method': 'dualp', 'presolve': False, 'tol_bnd': 1e-10, 'tol_dj': 1e-10}, {'maxit': 2**40, 'timeout': -5})
check('limits are clamped to what fits in a C int', o['it_lim'] == 2**31 - 1 and o['tm_lim'] == 0 and isinstance(o['it_lim'], int) and isinstance(o['tm_lim'], int), o)
check('options of the default settings', o['meth'] == 'GLP_DUALP' and o['presolve'] == 'GLP_OFF' and o['pricing'] == 'GLP_PT_PSE' and o['r_test'] == 'GLP_RT_HAR' and o['tol_bnd'] == 1e-10 and o['tol_piv'] == 1e-10 and o['msg_lev'] == 'GLP_MSG_OFF', o)

# ---- 4. The LP on which GLPK cycles when the problem is scaled
case = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lp_cycling_case.npz'))
cObj, cA, cb = case['obj'], case['A'], case['b']
OPT = -0.06005259929
t = time.time()
status, x, lp, used, _ = solve({'solver': 'glpk'}, cObj, cA, cb)
check('cycling LP, cvxopt binding: optimal at once, no fallback solve', status == 'optimal' and abs(float(cObj @ x.flatten()) - OPT) < 1e-8 and bool(np.all(cA @ x.flatten() <= cb + 1e-9)) and lp.lpFallbacks == 0 and used == {'cvxopt': 1, 'scikit': 0} and time.time() - t < 5.0, f'status {status}, {lp.lpFallbacks} fallbacks, {used}')
status, x, lp, used, _ = solve({'solver': 'glpk', 'glpkBinding': 'scikit'}, cObj, cA, cb)
check('cycling LP, scikit-glpk binding: optimal by the fallback solver', status == 'optimal' and abs(float(cObj @ x.flatten()) - OPT) < 1e-8 and lp.lpFallbacks == 1 and used == {'cvxopt': 0, 'scikit': 1}, f'status {status}, {lp.lpFallbacks} fallbacks, {used}')

# ---- 5. Which binding is used
obj, A, b = interiorLP(10, 2)
ref = float(obj @ solve({'solver': 'glpk'}, obj, A, b)[1].flatten())
def usedFor(lpopts, o=obj, a=A, bb=b):
    status, x, lp, used, wl = solve(lpopts, o, a, bb)
    return used, status, x
for ky, val in (('scale', False), ('init_basis', 'std'), ('basis_fac', 'luf+ft'), ('exact', False), ('ratio', 'flip')):
    used, status, x = usedFor({'solver': 'glpk', ky: val, 'fallback': None})
    check(f"lpopts with {ky!r}: scikit-glpk binding, optimal", used == {'cvxopt': 0, 'scikit': 1} and status == 'optimal' and abs(float(obj @ x.flatten()) - ref) < 1e-8, f'{used}, {status}')
encapsulateLP.glpkBinding = 'scikit'
used, status, x = usedFor({'solver': 'glpk', 'fallback': None})
check('module default set to scikit: scikit-glpk binding', used == {'cvxopt': 0, 'scikit': 1} and status == 'optimal', used)
used, status, x = usedFor({'solver': 'glpk', 'glpkBinding': 'cvxopt', 'fallback': None})
check("module default scikit, lpopts['glpkBinding'] = 'cvxopt': cvxopt binding", used == {'cvxopt': 1, 'scikit': 0} and status == 'optimal', used)
encapsulateLP.glpkBinding = 'cvxopt'
saved = encapsulateLP.cvxoptGlpk
encapsulateLP.cvxoptGlpk = None
used, status, x = usedFor({'solver': 'glpk', 'fallback': None})
check('cvxopt without GLPK: scikit-glpk binding', used == {'cvxopt': 0, 'scikit': 1} and status == 'optimal', used)
encapsulateLP.cvxoptGlpk = saved
# (GLPK itself does not reject non-finite data: through cvxopt it reports 'optimal' for an LP with a NaN. scikit-glpk
# refuses such an LP with a ValueError, which is what the solver 'glpk' has always done, so that stays as it is.)
for label, (o2, a2, b2) in (('a NaN entry in A', (obj, np.where(np.arange(A.size).reshape(A.shape) == 0, np.nan, A), b)), \
                           ('an infinite right-hand side', (obj, A, np.where(np.arange(b.size) == 0, np.inf, b))), \
                           ('a NaN in the objective', (np.where(np.arange(obj.size) == 0, np.nan, obj), A, b))):
    before = dict(calls)
    try:
        usedFor({'solver': 'glpk', 'fallback': None}, o2, a2, b2)
        outcome = 'returned'
    except ValueError:
        outcome = 'ValueError'
    check(f'LP with {label}: not given to the cvxopt binding (scikit-glpk: {outcome})', calls['cvxopt'] == before['cvxopt'] and calls['scikit'] == before['scikit'] + 1, f'{calls}, {before}')
# An LP without constraint rows: GLPK aborts the whole process on it, so runLP answers it without a solver, and
# the same way for every solver
noRowsA = np.zeros((0, 3)); noRowsb = np.zeros((0,))
for lpopts in ({'solver': 'glpk'}, {'solver': 'glpk', 'glpkBinding': 'scikit'}, {'solver': 'highs'}, {'solver': 'glpkCvxopt'}):
    before = dict(calls)
    status, x, lp, used, _ = solve(dict(lpopts), np.zeros(3), noRowsA, noRowsb)
    check(f'LP without rows, zero objective, {lpopts}: optimal at x = 0 without a solver', status == 'optimal' and x is not None and x.shape == (3, 1) and not np.any(x) and lp.lpFallbacks == 0 and lp.lpCount == 1 and calls == before, f'{status}, {x}, {used}')
    status, x, lp, used, _ = solve(dict(lpopts), np.array([0.0, -1.0, 0.0]), noRowsA, noRowsb)
    check(f'LP without rows, non-zero objective, {lpopts}: dual infeasible (unbounded), no fallback solve', status == 'dual infeasible' and x is None and lp.lpFallbacks == 0 and calls == before, f'{status}, {x}, {used}')
try:
    usedFor({'solver': 'glpk'}, np.array([0.0, np.nan, 0.0]), noRowsA, noRowsb)
    check('LP without rows, NaN in the objective: raises ValueError', False)
except ValueError:
    check('LP without rows, NaN in the objective: raises ValueError', True)
try:
    usedFor({'solver': 'nosuchsolver'}, np.zeros(3), noRowsA, noRowsb)
    check('LP without rows, unknown solver: still raises ValueError', False)
except ValueError as e:
    check('LP without rows, unknown solver: still raises ValueError', 'unknown LP solver' in str(e), str(e))
try:
    usedFor({'solver': 'glpk', 'glpkBinding': 'nosuchbinding'})
    check('unknown binding raises ValueError', False)
except ValueError as e:
    check('unknown binding raises ValueError', 'unknown GLPK binding' in str(e), str(e))
# column vectors and row counts as the callers pass them
used, status, x = usedFor({'solver': 'glpk', 'fallback': None}, obj.reshape(-1, 1), A, b.reshape(-1, 1))
check('objective and right-hand side as column vectors: cvxopt binding, optimal, x is a column', used == {'cvxopt': 1, 'scikit': 0} and status == 'optimal' and x.shape == (A.shape[1], 1) and abs(float(obj @ x.flatten()) - ref) < 1e-8, f'{used}, {status}')

# ---- 6. Through region_helpers, and the time per LP
n = 4
H = np.vstack([np.hstack([np.zeros((n, 1)), np.eye(n)]), np.hstack([np.ones((n, 1)), -np.eye(n)])])
before = dict(calls)
pt = region_helpers.findInteriorPoint(H, solver='glpk', lpObj=encapsulateLP.encapsulateLP())
check('interior point of the cube through region_helpers: centre, by the cvxopt binding', pt is not None and np.allclose(pt.flatten(), 0.5) and calls['cvxopt'] - before['cvxopt'] == 1 and calls['scikit'] == before['scikit'], f'{None if pt is None else pt.flatten()}')
print()
for (m, d) in [(10, 2), (25, 3), (60, 3), (150, 5)]:
    lps = [interiorLP(m, d) for _ in range(200)]
    line = f'   {lps[0][1].shape[0]:4d} rows x {d + 1} columns:'
    for binding in ('scikit', 'cvxopt'):
        lp = encapsulateLP.encapsulateLP()
        t = time.perf_counter()
        for (obj, A, b) in lps: lp.runLP(obj, A, b, lpopts={'solver': 'glpk', 'glpkBinding': binding, 'fallback': None})
        line += f'  {binding} {1e3 * (time.perf_counter() - t) / len(lps):.3f} ms per LP'
    print(line)

print(f'\n{"ALL TESTS PASSED" if len(failures) == 0 else str(len(failures)) + " TEST(S) FAILED"}')
sys.exit(0 if len(failures) == 0 else 1)
