"""Description of nodes consisting of a single-sided cavity with an atom
inside"""
import sympy
from sympy import symbols, sqrt
from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import Destroy, LocalSigma
from qnet.algebra.circuit_algebra import SLH, identity_matrix


def dagger(op):
    return op.adjoint()


def syms_ops(node_index, n_cavity):
    """Define symbols and operators for a single node, required to write the
    SLH for a single node"""
    HilAtom = LocalSpace('q%d' % int(node_index), basis=('g', 'e'),
                         order_index=(2*node_index))
    HilCavity = LocalSpace('c%d' % int(node_index), dimension=n_cavity,
                           order_index=(2*node_index+1))
    Sym = {}
    Sym['Delta'] = symbols(r'Delta_%s' % node_index, real=True)
    Sym['g'] = symbols(r'g_%s' % node_index, positive=True)
    Sym['Omega'] = symbols(r'Omega_%s' % node_index)
    Sym['I'] = sympy.I
    Sym['kappa'] = symbols(r'kappa', positive=True)
    Op = {}
    Op['a'] = Destroy(hs=HilCavity)
    Op['|g><g|'] = LocalSigma('g', 'g', hs=HilAtom)
    Op['|e><e|'] = LocalSigma('e', 'e', hs=HilAtom)
    Op['|e><g|'] = LocalSigma('e', 'g', hs=HilAtom)
    return Sym, Op


def node_hamiltonian(Sym, Op):
    """Symbolic Hamiltonian for a single node, in the RWA"""
    # Symbols
    Δ, g, Ω, I = (Sym['Delta'], Sym['g'], Sym['Omega'], Sym['I'])
    δ = g**2 / Δ
    # Cavity operators
    Op_a = Op['a']; Op_a_dag = dagger(Op_a); Op_n = Op_a_dag * Op_a
    # Qubit operators
    Op_gg = Op['|g><g|']; Op_eg = Op['|e><g|']; Op_ge = dagger(Op_eg)
    # Hamiltonian
    H = -δ * Op_n + (g**2/Δ) * Op_n * Op_gg \
        -I * (g / (2*Δ)) * Ω * (Op_eg*Op_a - Op_ge*Op_a_dag)
    return H


def node_slh(node_index, n_cavity):
    """SLH description for a single node with the given `node_index` (which
    will become the subscript in all symbols) and `n_cavity` number of levels
    for the cavity
    """
    Sym, Op = syms_ops(node_index, n_cavity)
    S = identity_matrix(1)
    κ = Sym['kappa']
    L = [sqrt(2 * κ) * Op['a'], ]
    H = node_hamiltonian(Sym, Op)
    return SLH(S, L, H)

