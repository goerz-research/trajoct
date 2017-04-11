"""Construct a network of nodes, where each node is as in
`crossed_sided_node.py`"""
from . import crossed_cavity_node_v1 as crossed_cavity_node
from . import double_sided_network_v1 as double_sided_network


def network_circuit(n_nodes, topology='open'):
    """See double_sided_network.network_circuit"""
    return double_sided_network.network_circuit(n_nodes, topology)


def network_slh(n_cavity, n_nodes, topology='open'):
    """See double_sided_network.nework_slh"""
    slh = double_sided_network.network_slh(
        n_cavity, n_nodes, topology=topology,
        _network_circuit=network_circuit,
        _node_slh=crossed_cavity_node.node_slh)
    return slh