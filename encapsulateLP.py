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
# two identical rows did, see tests/test_lp_cycling.py). So every solve gets a generous limit, and a solve that
# ends without an answer (iteration limit, numerical instability, ...) is repeated without scaling, and then with
# HiGHS if SciPy is available. None of this changes the result of a solve that GLPK finishes normally.
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

class encapsulateLP():

    def __init__(self):
        self.initializedSolvers = {}
        self.lpCount = 0

    def initSolver(self, solver='glpk', opts={}):
        self.lpCount = 0
        if solver == 'clp' and not 'clp' in self.initializedSolvers:
            self.d = opts['dim']
            self.cylp = CyClpSimplex()
            self.xVar = self.cylp.addVariable('x', self.d)
            self.cylp.logLevel = 0
            self.initializedSolvers['clp'] = True
        return self

    def runLP(self,obj,A,b,Ae=None,be=None,lpopts={'solver':'clp', 'fallback':{'solver':'glpk'}},msgID=''):
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
            def solveGLPK(opts):
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
                            **opts \
                        )
                xRes = np.array(res['x']).reshape(-1,1) if 'x' in res else None
                if 'status' in res:
                    statusRes = glpkStatus[res['status']]
                elif 'ret_code' in res:
                    statusRes = glpkRetCodes[res['ret_code']]
                else:
                    statusRes = 'unk'
                return statusRes, xRes
            conclusive = ('optimal', 'primal infeasible', 'dual infeasible')
            status, x = solveGLPK(glpkOpts)
            if not status in conclusive and glpkOpts.get('scale', True):
                # e.g. 'iteration limit exceeded' (cycling): the same LP is often solved at once without scaling
                self.lpRetries = getattr(self, 'lpRetries', 0) + 1
                status, x = solveGLPK(dict(glpkOpts, scale=False))
            if not status in conclusive and scipyLinprog is not None:
                self.lpRetries = getattr(self, 'lpRetries', 0) + 1
                try:
                    res = scipyLinprog(np.asarray(obj, dtype=np.float64).flatten(), A_ub=np.asarray(A, dtype=np.float64), b_ub=np.asarray(b, dtype=np.float64).flatten(), bounds=[(None, None)] * A.shape[1], method='highs')
                    if res.status == 0:
                        status, x = 'optimal', np.array(res.x).reshape(-1,1)
                    elif res.status == 2:
                        status, x = 'primal infeasible', None
                    elif res.status == 3:
                        status, x = 'dual infeasible', None
                except Exception:
                    pass

        if status != 'optimal' and status != 'primal infeasible' and status != 'dual infeasible':
            if 'fallback' in lpopts:
                lpopts = lpopts['fallback']
                status, x = self.runLP(obj,A,b,Ae,be,lpopts,msgID)
            else:
                status = 'unk'
                x = None
        return status, x
