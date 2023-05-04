import cvxopt
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from glpk import glpk, GLPK
import numpy as np
from copy import copy, deepcopy

glpkDefaults = {'basis_fac':'btf+cgr'}
glpkSimplexDefaults = {'init_basis':'adv','method':'dualp','tol_bnd':1e-10,'presolve':False}
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
        if solver == 'clp' and not 'clp' in self.initializedSolvers:
            self.d = opts['dim']
            self.cylp = CyClpSimplex()
            self.xVar = self.cylp.addVariable('x', self.d)
            self.cylp.logLevel = 0
            self.initializedSolvers['clp'] = True

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
            res = glpk( \
                        obj, \
                        A_ub=A, \
                        b_ub=b, \
                        message_level=GLPK.GLP_MSG_OFF, \
                        disp=False, \
                        simplex_options=simplexOpts, \
                        **glpkOpts \
                    )
            if 'x' in res:
                return glpkStatus[res['status']], res['x'].reshape(-1,1)
            elif 'status' in res:
                return glpkRetCodes[res['status']], None
            else:
                return 'unk', None

        if status != 'optimal' and status != 'primal infeasible' and status != 'dual infeasible':
            if 'fallback' in lpopts:
                lpopts = lpopts['fallback']
                status, x = self.runLP(obj,A,b,Ae,be,lpopts,msgID)
            else:
                status = 'unk'
                x = None
        return status, x
