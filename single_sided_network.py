"""Construct a network of nodes, where each node is as in
`single_sided_node.py`"""
from sympy import Symbol, symbols, sqrt
from qnet.algebra.circuit_algebra import (
    connect, CircuitSymbol, SLH, move_drive_to_H)
from qnet.circuit_components.beamsplitter_cc import Beamsplitter
from qnet.circuit_components.displace_cc import Displace

import single_sided_node


def node_symbol(node_index):
    """Symbolic node in the circuit"""
    return CircuitSymbol("Node_%d" % node_index, cdim=1)


def network_circuit(n_nodes, topology='open'):
    """Construct the network with the given topology"""
    if topology not in ['open', 'bs_fb', 'driven_bs_fb', 'driven_open']:
        raise ValueError("Unknown topology: %s" % topology)
    nodes = []
    connections = []
    prev_node = None
    for i in range(n_nodes):
        ind = i + 1
        cur_node = node_symbol(ind)
        nodes.append(cur_node)
        if prev_node is not None:
            connections.append(((prev_node, 0), (cur_node, 0)))
        prev_node = cur_node
    if 'bs_fb' in topology:
        BS = Beamsplitter('BS', theta=symbols('theta', real=True))
        nodes.append(BS)
        connections.append(((prev_node, 0), (BS, 0)))
        connections.append(((BS, 1), (nodes[0], 0)))
        if 'driven' in topology:
            W = Displace('W', alpha=symbols('alpha'))
            nodes.append(W)
            connections.append(((W, 0), (BS, 1)))
    else: # open topology
        if 'driven' in topology:  # driven_open
            W = Displace('W', alpha=symbols('alpha'))
            nodes.append(W)
            connections.append(((W, 0), (nodes[0], 0)))

    circuit = connect(nodes, connections, force_SLH=False)
    return circuit


def network_slh(n_cavity, n_nodes, topology='open', inhom=False):
    """Return the symbolic SLH for the entire network"""
    circuit = network_circuit(n_nodes, topology)
    slh_mapping = {}
    for i in range(n_nodes):
        ind = i + 1  # 1-based indexing of nodes
        slh_mapping[node_symbol(ind)] \
            = single_sided_node.node_slh(ind, n_cavity)
    S, L, H = circuit.substitute(slh_mapping).toSLH()
    κ = Symbol('kappa', positive=True)
    α, Ω = symbols('alpha Omega_alpha')
    S.expand().simplify_scalar()
    H = H.expand().simplify_scalar().substitute({
        α: (Ω.conjugate() / sqrt(κ)),
    })
    L = L.expand().simplify_scalar().substitute({
        α: (Ω.conjugate() / sqrt(κ)),
    })
    slh = SLH(S, L, H)
    if inhom:
        return slh
    else:
        return move_drive_to_H(slh)
