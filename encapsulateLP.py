import cvxopt
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from glpk import glpk, GLPK
import numpy as np
from copy import copy, deepcopy

try:
    from scipy.optimize import linprog as scipyLinprog
except ImportError:
    scipyLinprog = None

try:
    # GLPK through cvxopt's C binding (only if cvxopt was built with GLPK)
    from cvxopt import glpk as cvxoptGlpk
except ImportError:
    cvxoptGlpk = None

glpkDefaults = {'basis_fac':'btf+cgr'}
# GLPK's simplex has no iteration limit by default, and on a degenerate LP it can cycle forever (a 19 x 5 LP with
# two identical rows did, see tests/test_lp_cycling.py). So every GLPK solve gets a generous limit (unless
# 'maxit' is given in lpopts); a solve that stops at the limit is inconclusive, and goes to the fallback solver.
glpkMaxItBase = 20000
glpkMaxItPerRow = 50
glpkSimplexDefaults = {'init_basis':'adv','method':'dualp','tol_bnd':1e-10,'tol_dj':1e-10,'presolve':False}
glpkSimplexKeys = {'method','init_basis','steep','ratio','tol_bnd','tol_dj','tol_piv','obj_ll','obj_ul','presolve','exact'}
glpkArgs = {'scale','maxit','timeout','basis_fac'}
glpkStatus = {1:'undefined',2:'feasible',3:'infeasible',4:'primal infeasible',5:'optimal',6:'dual infeasible'}

# The solver 'glpk' can call GLPK in two ways ("bindings"):
#
#   'scikit'   scikit-glpk (glpk.glpk). It builds the problem through ctypes, which costs about half a millisecond
#              per LP whatever its size; for the LPs of a hyperplane arrangement (tens of rows, a handful of
#              columns) that is nearly all of the time of a solve.
#   'cvxopt'   cvxopt's C binding (cvxopt.glpk.lp). The problem is handed over as arrays: 0.03 ms for such an LP.
#
# Both run the same GLPK simplex with the same method, pricing, ratio test, tolerances, presolve setting and
# iteration/time limits, and both report optimal / primal infeasible / dual infeasible, or else an inconclusive
# status that sends the LP to the fallback solver (see runLP). The cvxopt binding cannot scale the problem,
# start from an advanced basis, choose the basis factorization ('scale', 'init_basis', 'basis_fac'; scikit-glpk
# does all three, the last two with glpkSimplexDefaults/glpkDefaults), use the long-step ratio test or solve in
# exact arithmetic. So it can end at a different optimal vertex when the optimum is not unique.
#
# glpkBinding is the binding that is used when lpopts has no 'glpkBinding' entry. The scikit binding is used
# regardless whenever the other one cannot do what is asked: lpopts names one of glpkScikitOnlyKeys (or the
# ratio test 'flip'), the problem has a non-finite entry, or cvxopt has no GLPK. (A problem without rows reaches
# neither: see runLP.)
glpkBinding = 'cvxopt' if cvxoptGlpk is not None else 'scikit'
glpkBindings = {'cvxopt','scikit'}
glpkScikitOnlyKeys = {'init_basis','exact','scale','basis_fac'}
# (GLPK aborts the process if a limit does not fit in a C int)
glpkIntMax = 2**31 - 1

def cvxoptGlpkOptions(simplexOpts,glpkOpts):
    # The options of scikit-glpk (simplexOpts: its simplex_options; glpkOpts: its keyword arguments), with its
    # defaults for what is not given, in the form cvxopt.glpk.lp wants them. cvxopt replaces a value it does not
    # recognize by GLPK's default (with a RuntimeWarning only), hence the explicit types.
    opts = { \
            'msg_lev':'GLP_MSG_OFF', \
            'meth':{'primal':'GLP_PRIMAL','dual':'GLP_DUAL','dualp':'GLP_DUALP'}[simplexOpts.get('method','primal')], \
            'pricing':('GLP_PT_PSE' if simplexOpts.get('steep',True) else 'GLP_PT_STD'), \
            'r_test':{'relax':'GLP_RT_HAR','norelax':'GLP_RT_STD'}[simplexOpts.get('ratio','relax')], \
            'tol_bnd':float(simplexOpts.get('tol_bnd',1e-7)), \
            'tol_dj':float(simplexOpts.get('tol_dj',1e-7)), \
            'tol_piv':float(simplexOpts.get('tol_piv',1e-10)), \
            'presolve':('GLP_ON' if simplexOpts.get('presolve',True) else 'GLP_OFF'), \
            'it_lim':int(min(max(glpkOpts.get('maxit',glpkIntMax),0),glpkIntMax)), \
            'tm_lim':int(min(max(glpkOpts.get('timeout',glpkIntMax),0),glpkIntMax)) \
        }
    # (scikit-glpk only sets these limits if they are "true", i.e. given and non-zero)
    for ky in ['obj_ll','obj_ul']:
        if simplexOpts.get(ky,False):
            opts[ky] = float(simplexOpts[ky])
    return opts
glpkRetCodes = { \
        0: 'LP problem instance has been successfully solved', \
        1: 'invalid basis', \
        2: 'singular matrix', \
        3: 'ill-conditioned matrix', \
        4: 'invalid bounds', \
        5: 'solver failed', \
        6: 'objective lower limit reached', \
        7: 'objective upper limit reached', \
        8: 'iteration limit exceeded', \
        9: 'time limit exceeded', \
        10: 'primal infeasible', \
        11: 'dual infeasible', \
        12: 'root LP optimum not provided', \
        13: 'search terminated by application', \
        14: 'relative mip gap tolerance reached', \
        15: 'no primal/dual feasible solution', \
        16: 'no convergence', \
        17: 'numerical instability', \
        18: 'invalid data', \
        19: 'result out of range' \
}

# HiGHS (through scipy.optimize.linprog). The defaults mirror those of GLPK above: the same feasibility tolerances
# (1e-10 is the smallest HiGHS accepts) and no presolve, so that a solve ends with a definite status (with presolve
# HiGHS can stop at "infeasible or unbounded").
highsDefaults = {'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'presolve':False}
highsKeys = {'primal_feasibility_tolerance','dual_feasibility_tolerance','presolve','maxiter','time_limit','simplex_dual_edge_weight_strategy'}
# (scipy.optimize.linprog status codes; 1 = iteration/time limit and 4 = numerical difficulties are inconclusive)
highsStatus = {0:'optimal',1:'iteration limit exceeded',2:'primal infeasible',3:'dual infeasible',4:'numerical instability'}

knownSolvers = {'glpkCvxopt','clp','glpk','highs'}
conclusiveStatus = ('optimal','primal infeasible','dual infeasible')
# The fallback solver of a solver whose lpopts have no 'fallback' key (see runLP)
defaultFallback = {'glpk':{'solver':'highs'},'highs':{'solver':'glpk'}}

class encapsulateLP():

    def __init__(self):
        self.initializedSolvers = {}
        self.lpCount = 0
        self.lpFallbacks = 0

    def initSolver(self, solver='glpk', opts={}):
        self.lpCount = 0
        self.lpFallbacks = 0
        if solver == 'clp' and not 'clp' in self.initializedSolvers:
            self.d = opts['dim']
            self.cylp = CyClpSimplex()
            self.xVar = self.cylp.addVariable('x', self.d)
            self.cylp.logLevel = 0
            self.initializedSolvers['clp'] = True
        return self

    # lpopts: {'solver': 'glpk' | 'highs' | 'clp' | 'glpkCvxopt', <options of that solver>, 'fallback': lpopts}
    # (options: glpkSimplexKeys, glpkArgs and 'glpkBinding' ('cvxopt' | 'scikit', see glpkBinding above) for GLPK;
    # highsKeys and 'highs_method', a linprog method, for HiGHS)
    # If the solver ends without an answer (anything but optimal / primal infeasible / dual infeasible), the LP is
    # solved again with lpopts['fallback'] (itself an lpopts dictionary, so fallbacks can be chained). Without a
    # 'fallback' key, the fallback of GLPK is HiGHS and that of HiGHS is GLPK, each with its default options and
    # each tried at most once per LP; 'fallback':None disables it. self.lpFallbacks counts the fallback solves.
    def runLP(self,obj,A,b,Ae=None,be=None,lpopts={'solver':'clp', 'fallback':{'solver':'glpk'}},msgID='',_tried=()):
        self.lpCount += 1
        if np.shape(A)[0] == 0 and lpopts['solver'] in knownSolvers:
            # An LP without constraints needs no solver, and must not reach one: GLPK aborts the whole process on a
            # problem without rows (no exception, no traceback), through either binding. Without constraints every
            # x is optimal for a zero objective (x = 0 is returned, which is also what HiGHS and CLP answer), and
            # any other objective is unbounded.
            objArr = np.asarray(obj,dtype=np.float64).flatten()
            if not np.all(np.isfinite(objArr)):
                raise ValueError(f'{msgID}: the objective of an LP must be finite')
            if np.any(objArr != 0):
                return 'dual infeasible', None
            return 'optimal', np.zeros((objArr.size,1),dtype=np.float64)
        if lpopts['solver']=='glpkCvxopt':
            cvxArgs = [cvxopt.matrix(obj), cvxopt.matrix(A), cvxopt.matrix(b)]
            sol = cvxopt.solvers.lp(*cvxArgs,solver='glpk',options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
            status = sol['status']
            x = sol['x']
        elif lpopts['solver']=='clp':
            assert 'clp' in self.initializedSolvers and self.initializedSolvers['clp'], f'{msgID}: Please initialize solver CLP first'
            for constr in range(len(self.cylp.constraints)):
                self.cylp.removeConstraint(self.cylp.constraints[constr].name)
            self.cylp += np.matrix(A) * self.xVar <= CyLPArray(b.flatten())
            self.cylp.objective = CyLPArray(obj)
            try:
                status = self.cylp.primal()
                x = np.array(self.cylp.primalVariableSolution['x']).reshape((self.d,1))
            except:
                # Something went wrong with the CLP solver, so force use of GLPK
                print(' ')
                print('********************  PE' + msgID + ' WARNING!!  ********************')
                print('PE' + msgID + ': Trying to execute fallback solvers in seqeunce...' )
                print(' ')
                status = 'unk'
        elif lpopts['solver'] == 'glpk':
            simplexOpts = copy(glpkSimplexDefaults)
            for ky in lpopts.keys():
                if ky in glpkSimplexKeys:
                    simplexOpts[ky] = lpopts[ky]
            glpkOpts = copy(glpkDefaults)
            for ky in lpopts.keys():
                if ky in glpkArgs:
                    glpkOpts[ky] = lpopts[ky]
            if not 'maxit' in glpkOpts:
                glpkOpts['maxit'] = glpkMaxItBase + glpkMaxItPerRow * (A.shape[0] + A.shape[1])
            binding = lpopts.get('glpkBinding',glpkBinding)
            if not binding in glpkBindings:
                raise ValueError(f'{msgID}: unknown GLPK binding {binding!r} (known bindings: {sorted(glpkBindings)})')
            if binding == 'cvxopt':
                # (see glpkBinding above for when the cvxopt binding cannot be used)
                objC = np.ascontiguousarray(obj,dtype=np.float64).reshape(-1,1)
                AC = np.ascontiguousarray(A,dtype=np.float64)
                bC = np.ascontiguousarray(b,dtype=np.float64).reshape(-1,1)
                if cvxoptGlpk is None or any([ky in lpopts for ky in glpkScikitOnlyKeys]) or simplexOpts.get('ratio','relax') == 'flip' \
                        or AC.ndim != 2 or AC.shape[0] < 1 or AC.shape[1] < 1 or AC.shape != (bC.shape[0],objC.shape[0]) \
                        or not (np.all(np.isfinite(AC)) and np.all(np.isfinite(bC)) and np.all(np.isfinite(objC))):
                    binding = 'scikit'
            if binding == 'cvxopt':
                status, x, _ = cvxoptGlpk.lp(cvxopt.matrix(objC),cvxopt.matrix(AC),cvxopt.matrix(bC),options=cvxoptGlpkOptions(simplexOpts,glpkOpts))
                # ('unknown' is everything else, e.g. the iteration limit: inconclusive, like the other codes of GLPK)
                x = np.array(x,dtype=np.float64).reshape(-1,1) if (status == 'optimal' and x is not None) else None
            else:
                res = glpk( \
                            obj, \
                            A_ub=A.copy(), \
                            b_ub=b.copy(), \
                            A_eq=None, \
                            b_eq=None, \
                            bounds=(None, None), \
                            message_level=GLPK.GLP_MSG_OFF, \
                            disp=False, \
                            simplex_options=simplexOpts, \
                            **glpkOpts \
                        )
                if 'x' in res:
                    x = np.array(res['x']).reshape(-1,1)
                else:
                    x = None
                if 'status' in res:
                    status = glpkStatus[res['status']]
                elif 'ret_code' in res:
                    status = glpkRetCodes[res['ret_code']]
                else:
                    status = 'unk'
        elif lpopts['solver'] == 'highs':
            if scipyLinprog is None:
                raise ImportError(f'{msgID}: solver \'highs\' requires SciPy (scipy.optimize.linprog)')
            highsOpts = copy(highsDefaults)
            for ky in lpopts.keys():
                if ky in highsKeys:
                    highsOpts[ky] = lpopts[ky]
            x = None
            try:
                res = scipyLinprog( \
                            np.asarray(obj,dtype=np.float64).flatten(), \
                            A_ub=np.asarray(A,dtype=np.float64), \
                            b_ub=np.asarray(b,dtype=np.float64).flatten(), \
                            bounds=(None, None), \
                            method=lpopts.get('highs_method','highs'), \
                            options=highsOpts \
                        )
                status = highsStatus.get(res.status,'unk')
                if status == 'optimal':
                    x = np.array(res.x).reshape(-1,1)
            except ValueError as e:
                # linprog rejects e.g. NaN/inf coefficients; like a GLPK failure, this is an inconclusive solve
                status = f'invalid input ({e})'
        else:
            raise ValueError(f'{msgID}: unknown LP solver {lpopts["solver"]!r} (known solvers: {sorted(knownSolvers)})')

        if not status in conclusiveStatus:
            tried = _tried + (lpopts['solver'],)
            if 'fallback' in lpopts:
                fallback = lpopts['fallback']
                if isinstance(fallback,str):
                    # (some callers pass the name of the fallback solver instead of an lpopts dictionary)
                    fallback = {'solver':fallback}
            else:
                # No fallback specified: use the "other" one of GLPK and HiGHS, unless it was already tried on
                # the way here. ('fallback':None in lpopts means no fallback at all.)
                fallback = defaultFallback.get(lpopts['solver'],None)
                if fallback is not None and (fallback['solver'] in tried or (fallback['solver'] == 'highs' and scipyLinprog is None)):
                    fallback = None
            if fallback is not None:
                self.lpFallbacks += 1
                status, x = self.runLP(obj,A,b,Ae,be,fallback,msgID,_tried=tried)
            else:
                status = 'unk'
                x = None
        return status, x
