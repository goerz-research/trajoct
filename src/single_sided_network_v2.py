from . import single_sided_node_v1 as single_sided_node
from qnet.algebra import Matrix, SLH, OperatorPlus
from sympy import sqrt, I, symbols


def dagger(op):
    return op.adjoint()


def network_slh(n_cavity, n_nodes, topology='open', inhom=False):
    """Return the symbolic SLH for the entire network"""
    if topology != 'open':
        raise NotImplementedError()
    L = 0
    Hs = []
    syms = []
    ops = []
    κ = symbols(r'kappa', positive=True)
    for i in range(n_nodes):
        ind = i + 1  # 1-based indexing of nodes
        syms_i, ops_i = single_sided_node.syms_ops(
            node_index=ind, n_cavity=n_cavity)
        Hs.append(single_sided_node.node_hamiltonian(syms_i, ops_i))
        syms.append(syms_i)
        ops.append(ops_i)
    H_terms = []
    L_terms = []
    for (i, H_i) in enumerate(Hs):
        L_terms.append(sqrt(2 * κ) * ops[i]['a'])
        H_terms.append(H_i)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            assert syms[i]['kappa'] == syms[j]['kappa'] == κ
            H_ij = I * κ * dagger(ops[i]['a']) * ops[j]['a']
            H_terms.append(H_ij + dagger(H_ij))
    H = OperatorPlus(*H_terms)
    L = OperatorPlus(*L_terms)
    slh = SLH([[1]], Matrix([[L]]), H)
    if not inhom:
        pass  # by construction, we never have an inhomogenity
    slh.n_cavity = n_cavity
    slh.n_nodes = n_nodes
    return slh
