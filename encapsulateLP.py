import cvxopt
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import numpy as np

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

    def runLP(self,obj,A,b,Ae=None,be=None,lpopts={'solver':'clp', 'fallback':'glpk'},msgID=''):
        self.lpCount += 1
        if lpopts['solver']=='glpk':
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
                print('PE' + msgID + ': Needed to fallback to GLPK for unknown reasons!!' )
                print(' ')
                status = 'unk'

        if status != 'optimal' and status != 'primal infeasible' and status != 'dual infeasible':
            if 'fallback' in lpopts and lpopts['fallback'] != lpopts['solver']:
                lpopts['solver'] = lpopts['fallback']
                lpopts.pop('fallback',None)
                status, x = self.runLP(obj,A,b,Ae,be,lpopts,msgID)
            else:
                status = 'unk'
                x = None
        return status, x
