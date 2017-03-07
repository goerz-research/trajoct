import numpy as np
import pandas as pd
from collections import OrderedDict
import QDYN

def get_weyl_table(U_of_t_dat):
    tgrid = np.genfromtxt(U_of_t_dat, usecols=(0, ))
    concurrence = []
    loss = []
    c1s = []
    c2s = []
    c3s = []
    for U in QDYN.prop_gate.get_prop_gate_of_t(U_of_t_dat):
        U_closest_unitary = U.closest_unitary()
        concurrence.append(U_closest_unitary.concurrence())
        loss.append(U.pop_loss())
        c1, c2, c3 = U_closest_unitary.weyl_coordinates()
        c1s.append(c1)
        c2s.append(c2)
        c3s.append(c3)
    return pd.DataFrame(data=OrderedDict([
        ('t [microsec]', tgrid),
        ('concurrence', concurrence),
        ('loss', loss),
        ('c1', c1s),
        ('c2', c1s),
        ('c3', c3s),
    ]))


