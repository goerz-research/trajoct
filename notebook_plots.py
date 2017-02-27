"""Plotting routines for inside the notebook"""
from os.path import join
from glob import glob
from collections import OrderedDict

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from IPython.display import display, Latex
from qnet.printing import tex

import QDYN

from algebra import split_hamiltonian


def show_summary(rf, pulses='pulse*.oct.dat'):
    """Show plot of observables"""
    fig = plt.figure(figsize=(16, 3.5), dpi=70)

    ax = fig.add_subplot(131)
    try:
        render_population(ax, rf)
    except OSError:
        pass

    ax = fig.add_subplot(132)
    try:
        render_excitation(ax, rf)
    except OSError:
        pass

    ax = fig.add_subplot(133)
    try:
        render_pulses(ax, rf, pulses)
    except OSError:
        pass

    plt.show(fig)


def show_dicke_summary(rf, pulses='pulse*.oct.dat'):
    """Show plot of observables"""
    fig = plt.figure(figsize=(16, 3.5), dpi=70)

    ax = fig.add_subplot(121)
    try:
        render_excitation(ax, rf, atoms_only=True)
    except OSError:
        pass

    ax = fig.add_subplot(122)
    try:
        render_pulses(ax, rf, pulses)
    except OSError:
        pass

    plt.show(fig)


def render_pulses(ax, rf, pulses='pulse*.oct.dat'):
    for i, pulse_file in enumerate(sorted(glob(join(rf, pulses)))):
        p = QDYN.pulse.Pulse.read(pulse_file)
        p.render_pulse(ax, label='pulse %d' % (i+1))
    ax.legend()


def render_population(ax, rf):
    qubit_pop = np.genfromtxt(join(rf, 'qubit_pop.dat')).transpose()
    tgrid = qubit_pop[0]  # microsecond
    ax.plot(tgrid, qubit_pop[1], label=r'00')
    ax.plot(tgrid, qubit_pop[2], label=r'01')
    ax.plot(tgrid, qubit_pop[3], label=r'10')
    ax.plot(tgrid, qubit_pop[4], label=r'11')
    total = qubit_pop[1] + qubit_pop[2] + qubit_pop[3] + qubit_pop[4]
    ax.plot(tgrid, total, label=r'total', ls='--')
    ax.legend(loc='best', fancybox=True, framealpha=0.5)
    ax.set_ylim([0, 1.1])
    ax.set_xlabel("time (microsecond)")
    ax.set_ylabel("population")


def render_excitation(ax, rf, atoms_only=False):
    exc_file = join(rf, 'excitation.dat')
    with open(exc_file) as in_fh:
        header = in_fh.readline()
        labels = header.strip('#').strip().split()[2:]
    excitation = np.genfromtxt(exc_file).transpose()
    tgrid = excitation[0]  # microsecond
    total = np.zeros(len(tgrid))
    accept = lambda label: True
    render_label = lambda label: label
    if atoms_only:
        accept = lambda label: label.startswith('q')
        rener_label = lambda label: label[1:]
    for i, label in enumerate(labels):
        total += excitation[i+1]
        if accept(label):
            ax.plot(tgrid, excitation[i+1], label=render_label(label))
    ax.plot(tgrid, total, ls='--', label='total')
    ax.legend(loc='best', fancybox=True, framealpha=0.5)
    ax.set_ylim([0, max(1.0, 1.05*ax.get_ylim()[1])])
    ax.set_xlabel("time (microsecond)")
    ax.set_ylabel("excitation")


def display_with_cc(expr):
    from qnet.algebra.abstract_algebra import extra_binary_rules
    from qnet.algebra.operator_algebra import OperatorPlus, create_operator_pm_cc
    with extra_binary_rules(OperatorPlus, create_operator_pm_cc()):
        display(expr.simplify())


def display_hamiltonian(H):
    """display all the terms of the given Hamiltonian, separating drift,
    interaction, and the various control Hamiltonian onto different lines"""
    terms = split_hamiltonian(H)

    def label(s):
        if s == 'H0':
            return r'\hat{H}_0'
        elif s == 'Hint':
            return r'\hat{H}_{\text{int}}'
        else:
            try:
                prefix, ind = s.split('_')
            except ValueError:
                print(s)
                raise
            return r'\hat{H}_{d_%s}' % ind

    lines = []
    lines.append(r'\begin{align}')
    lines.append(r'  \hat{H} &= %s\\' % " + ".join([label(name) for name in terms.keys()]))
    for name, H in terms.items():
        lines.append(r'  %s &= %s\\' % (label(name), tex(H)))
    lines.append(r'\end{align}')
    display(Latex("\n".join(lines)))


def get_weyl_table(U_of_t_dat):
    """Get a table of time, concurrence, loss, and weyl coordinates from the
    gates written out by the oct_prop_gate utility"""
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


def get_weyl_chamber(U_of_t_dat, range=None):
    w = QDYN.weyl.WeylChamber()
    w.fig_width = 20
    w.fig_height = 15
    if range is not None:
        i_min, i_max = range
    for i, U in enumerate(QDYN.prop_gate.get_prop_gate_of_t(U_of_t_dat)):
        if range is None or (i > i_min and i < i_max):
            w.add_gate(U.closest_unitary())
    return w
