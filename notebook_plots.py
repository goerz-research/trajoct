"""Plotting routines for inside the notebook"""
from os.path import join

import matplotlib.pylab as plt
import numpy as np
from IPython.display import display, Latex
from qnet.printing import tex

import QDYN

from algebra import split_hamiltonian


def show_observables(rf, logical_pops_file='qubit_pop.dat',
        beta_pops_file='beta_pop.dat', exc_file='cavity_excitation.dat',
        pulse1_file='pulse1.dat', pulse2_file='pulse2.dat'):
    """Show plot of observables"""
    fig = plt.figure(figsize=(16,3.5), dpi=70)

    qubit_pop = np.genfromtxt(join(rf, logical_pops_file)).transpose()
    beta_pop = np.genfromtxt(join(rf, beta_pops_file)).transpose()
    exc = np.genfromtxt(join(rf, exc_file)).transpose()
    p1 = QDYN.pulse.Pulse.read(join(rf, pulse1_file))
    p2 = QDYN.pulse.Pulse.read(join(rf, pulse2_file))

    ax = fig.add_subplot(131)
    tgrid = qubit_pop[0] # microsecond
    ax.plot(tgrid, qubit_pop[1], label=r'00')
    ax.plot(tgrid, qubit_pop[2], label=r'01')
    ax.plot(tgrid, qubit_pop[3], label=r'10')
    ax.plot(tgrid, qubit_pop[4], label=r'11')
    ax.plot(tgrid, beta_pop[1], label=r'0010')
    ax.plot(tgrid, beta_pop[2], label=r'0001')
    analytical_pop = qubit_pop[1] + qubit_pop[2] + qubit_pop[3] \
                     + beta_pop[1] + beta_pop[2]
    ax.plot(tgrid, analytical_pop, label=r'ana. subsp.')
    ax.legend(loc='best', fancybox=True, framealpha=0.5)
    ax.set_xlabel("time (microsecond)")
    ax.set_ylabel("population")

    ax = fig.add_subplot(132)
    ax.plot(tgrid, exc[1], label=r'<n> (cav 1)')
    ax.plot(tgrid, exc[2], label=r'<n> (cav 2)')
    ax.plot(tgrid, exc[3], label=r'<L>')
    ax.legend(loc='best', fancybox=True, framealpha=0.5)
    ax.set_xlabel("time (microsecond)")
    ax.set_ylabel("cavity excitation")

    ax = fig.add_subplot(133)
    p1.render_pulse(ax, label='pulse 1')
    p2.render_pulse(ax, label='pulse 2')
    ax.legend(loc='best', fancybox=True, framealpha=0.5)

    #ax.set_xlim(-4, -1)


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
