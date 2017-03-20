from collections import OrderedDict
from sympy import symbols, srepr

from qnet.algebra.abstract_algebra import extra_binary_rules
from qnet.algebra.operator_algebra import OperatorPlus, create_operator_pm_cc
from qnet.algebra.hilbert_space_algebra import LocalSpace, ProductSpace


def split_hamiltonian(H, use_cc=True, controls='Omega'):
    """Split the given symbolic Hamiltonian into drift, interaction, and drive
    Hamiltonians. Returns a dictionary with keys 'H0', 'Hint', 'Hd_1',
    'Hd_2', ..., mapping to the respective Hamiltonians.

    Args:
        H (Operator): The full (symbolic) Hamiltonian
        use_cc (bool): Whether to use '+ c.c' in the split Hamiltonian for
            easier readability
        controls (str or list): List of control symbols. If a string, every
            symbol whose name starts with that string is considered a control.
        """
    res = OrderedDict()
    H = H.expand().simplify_scalar()
    if isinstance(controls, str):
        controls = sorted(
                [sym for sym in H.all_symbols()
                 if sym.name.startswith(controls)],
                key=str)
    n_controls = len(controls)
    Hdrift = H.substitute({control: 0 for control in controls})
    res['H0'] = OperatorPlus.create(
            *[H for H in Hdrift.operands if isinstance(H.space, LocalSpace)])
    res['Hint'] = OperatorPlus.create(
            *[H for H in Hdrift.operands if isinstance(H.space, ProductSpace)]
            ).expand().simplify_scalar()
    Hdrive = (H - Hdrift).expand()

    def all_zero_except(controls, i):
        return {control: 0
                for (j, control) in enumerate(controls)
                if j != i}

    for i in range(n_controls):
        mapping = all_zero_except(controls, i)
        res['H_%s' % str(controls[i])] = Hdrive.substitute(mapping)
    if use_cc:
        for name, H in res.items():
            with extra_binary_rules(OperatorPlus, create_operator_pm_cc()):
                res[name] = H.simplify_scalar().simplify()
    return res


def generate_num_vals_code(syms, controls=None):
    """Generate the code of a dictionary that maps every symbols in `syms`
    (excluding `controls`) to a numerical value"""
    if controls is None:
        controls = []
    print("num_vals = {")
    for sym in syms:
        if sym not in controls:
            print("    %s: 0.0," % srepr(sym))
    print("}")
