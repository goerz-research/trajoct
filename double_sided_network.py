"""Construct a network of nodes, where each node is as in
`double_sided_node.py`"""
from qnet.algebra.circuit_algebra import connect, CircuitSymbol, SLH

import double_sided_node


def node_symbol(node_index):
    """Symbolic node in the circuit"""
    return CircuitSymbol("Node_%d" % node_index, cdim=2)


def network_circuit(n_nodes, topology='open'):
    """Construct the network with the given topology"""
    if topology not in ['open', 'bs_fb']:
        raise ValueError("Unknown topology: %s" % topology)
    nodes = []
    connections = []
    prev_node = None
    for i in range(n_nodes):
        ind = i + 1
        cur_node = CircuitSymbol("Node%d" % ind, cdim=2)
        nodes.append(cur_node)
        if prev_node is not None:
            if topology in [None, 'open', 'FB']:
                connections.append(((prev_node, 0), (cur_node, 0)))
                connections.append(((cur_node, 1), (prev_node, 0)))
            else:
                raise ValueError("Unknown topology: %s" % topology)
        prev_node = cur_node
    if topology == 'FB':
        connections.append(((cur_node, 0), (cur_node, 1)))
    circuit = connect(nodes, connections, force_SLH=False)
    return circuit


def network_slh(n_cavity, n_nodes, topology='open', _node_slh=None):
    """Set up a chain of JC system with two channels

    Args:
        n_cavity (int): Number of levels in the cavity (numerical truncation)
        n_nodes (int):  Number of nodes in the chain
        topology (str or None): How the nodes should be linked up, see below
        _node_slh (callable or None): routine that returns the SLH model for a
            single node. If None, `double_sided_node.node_slh`

    Notes:

        The `topology` can take the following values:
        * "open": chain is open-ended. The total system will have two
          I/O channels

              ->X->X->X->
              <-X<-X<-X<-

        * "FB": The rightmost output is fed back into the rightmost input::

              >-------+
                      |
              <-------+
    """
    if _node_slh is None:
        _node_slh = double_sided_node.node_slh
    circuit = network_circuit(n_nodes, topology)
    slh_mapping = {}
    for i in range(n_nodes):
        ind = i + 1  # 1-based indexing of nodes
        slh_mapping[node_symbol(ind)] = _node_slh(ind, n_cavity)
    S, L, H = circuit.substitute(slh_mapping).toSLH()
    slh = SLH(
        S.expand().simplify_scalar(),
        L.expand().simplify_scalar(),
        H.expand().simplify_scalar())
    return slh
