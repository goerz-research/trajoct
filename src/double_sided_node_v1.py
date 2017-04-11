"""Description of nodes consisting of a double-sided cavity with an atom
inside"""
from sympy import symbols, sqrt

from qnet.algebra.circuit_algebra import SLH, identity_matrix

from . import single_sided_node_v1 as single_sided_node


def syms_ops(node_index, n_cavity):
    """Define symbols and operators for a single node, required to write the
    SLH for a single node"""
    Sym, Op = single_sided_node.syms_ops(node_index, n_cavity)
    del Sym['kappa']
    Sym['kappa_l'] = symbols(r'kappa_l', positive=True)
    Sym['kappa_r'] = symbols(r'kappa_r', positive=True)
    return Sym, Op


def node_slh(node_index, n_cavity):
    """SLH description for a single node with the given `node_index` (which
    will become the subscript in all symbols) and `n_cavity` number of levels
    for the cavity
    """
    Sym, Op = syms_ops(node_index, n_cavity)
    S = identity_matrix(2)
    kappa_l = Sym['kappa_l']
    kappa_r = Sym['kappa_r']
    L = [sqrt(2*kappa_l) * Op['a'], sqrt(2*kappa_r) * Op['a']]
    H = single_sided_node.node_hamiltonian(Sym, Op)
    return SLH(S, L, H)
