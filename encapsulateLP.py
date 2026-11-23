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
    # (options: glpkSimplexKeys and glpkArgs for GLPK; highsKeys and 'highs_method', a linprog method, for HiGHS)
    # If the solver ends without an answer (anything but optimal / primal infeasible / dual infeasible), the LP is
    # solved again with lpopts['fallback'] (itself an lpopts dictionary, so fallbacks can be chained). Without a
    # 'fallback' key, the fallback of GLPK is HiGHS and that of HiGHS is GLPK, each with its default options and
    # each tried at most once per LP; 'fallback':None disables it. self.lpFallbacks counts the fallback solves.
    def runLP(self,obj,A,b,Ae=None,be=None,lpopts={'solver':'clp', 'fallback':{'solver':'glpk'}},msgID='',_tried=()):
        self.lpCount += 1
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
