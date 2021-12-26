import numpy as np
import encapsulateLP



# H2 is a CDD-style matrix specifying inequality constraints, and intIdx is a list of indices of inequalities to check for redundancy
# The return value is a list of indices into the list intIdx specifying which of those constraints are non-redundant
def lpMinHRep(H2,constraint_list_in,intIdx,solver='glpk',safe=False,lpObj=None):
    if len(intIdx) == 0:
        return np.full(0,0,dtype=bool)

    if lpObj is None:
       lpObj = encapsulateLP.encapsulateLP()
       lpObj.initSolver(solver=solver)

    restricted = False if constraint_list_in is None else True

    # H2 should be a view into the CDD-formatted H matrix selected by taking boolIdx or intIdx rows thereof
    if safe:
        H = H2 if not restricted else H2[constraint_list_in[0:len(H2)],:]
    else:
        # This version of H has an extra row, that we can use for the another constraint
        H = np.vstack([H2, [H2[0,:]] ]) if not restricted else np.vstack([H2[constraint_list_in,:], [H2[0,:]] ])

    to_keep = []
    constraint_list = np.full(len(H),True,dtype=bool)
    if restricted:
        restIdxs = np.nonzero(constraint_list_in)[0]
        offsetTab = dict(zip(restIdxs,range(len(restIdxs))))
    for idx in range(len(intIdx)):
        if restricted and (not constraint_list_in[intIdx[idx]]):
            continue
        offsetIdx = intIdx[idx] if not restricted else offsetTab[intIdx[idx]]
        # You can test the current hyperplane for consideration by other means at this point
        if not safe:
            # Set the extra row to the negation of the pre-relaxed current constraint
            H[-1,:] = -H2[intIdx[idx],:]
        H[offsetIdx,0] += 1
        status, x = lpObj.runLP( \
                H2[intIdx[idx],1:], \
                -H[constraint_list,1:], \
                H[constraint_list,0], \
                lpopts = {'solver':solver, 'fallback':'glpk'} if solver != 'glpk' else {'solver':'glpk'}, \
                msgID = 'None' \
            )
        H[offsetIdx,0] -= 1

        if status != 'optimal' and (safe or status != 'primal infeasible') and status != 'dual infeasible':
            print('********************  WARNING!!  ********************')
            print('Infeasible or numerical ill-conditioning detected at node -- RESULTS MAY NOT BE ACCURATE!!')
            return [set([]), 0]
        if (safe and -H2[intIdx[idx],1:]@x < H2[intIdx[idx],0]) \
            or (not safe and (status == 'primal infeasible' or np.all(-H2[intIdx[idx],1:]@x - H2[intIdx[idx],0] <= 1e-10))):
            # inequality is redundant, so skip it
            constraint_list[offsetIdx] = False
        else:
            to_keep.append(idx)

    return to_keep