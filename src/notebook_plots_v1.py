"""Plotting routines for inside the notebook"""
from os.path import join
from glob import glob
from collections import OrderedDict

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import sympy
from sympy import Symbol
from IPython.display import display, Latex
from qnet.printing import tex
from qnet.algebra import pattern, ScalarTimesOperator

import QDYN

from .algebra_v1 import split_hamiltonian
from .pulse_smoothing_v1 import pulse_delta_smoothing


SUBSCRIPT_MAPPING = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆',
    '7': '₇', '8': '₈', '9': '₉', '(': '₍', ')': '₎', '+': '₊', '-': '₋',
    '=': '₌', 'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ', 'h': 'ₕ', 'k': 'ₖ',
    'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'p': 'ₚ', 's': 'ₛ', 't': 'ₜ',
    'β': 'ᵦ', 'γ': 'ᵧ', 'ρ': 'ᵨ', 'φ': 'ᵩ', 'χ': 'ᵪ'
}


def show_summary_gate(
        rf, pulses='pulse*.oct.dat', single_node=False, xrange=None):
    """Show plot of observables"""
    fig = plt.figure(figsize=(16, 3.5), dpi=70)

    axs = []

    axs.append(fig.add_subplot(131))
    try:
        render_population(axs[-1], rf, single_node=single_node)
    except OSError:
        pass

    axs.append(fig.add_subplot(132))
    try:
        render_excitation(axs[-1], rf)
    except OSError:
        pass

    axs.append(fig.add_subplot(133))
    try:
        render_pulses(axs[-1], rf, pulses)
    except OSError:
        pass

    if xrange is not None:
        for ax in axs:
            ax.set_xlim(*xrange)

    plt.show(fig)


def show_summary_dicke(
        rf, pulses='pulse*.oct.dat', single_node=False, xrange=None, dpi=70):
    """Show plot of observables"""
    fig = plt.figure(figsize=(16, 3.5), dpi=dpi)

    axs = []

    axs.append(fig.add_subplot(131))
    try:
        render_excitation(axs[-1], rf, filter='c')
    except OSError:
        pass

    axs.append(fig.add_subplot(132))
    try:
        render_excitation(axs[-1], rf, filter='q')
    except OSError:
        pass

    axs.append(fig.add_subplot(133))
    try:
        render_pulses(axs[-1], rf, pulses)
    except OSError:
        pass

    if xrange is not None:
        for ax in axs:
            ax.set_xlim(*xrange)

    plt.show(fig)


def render_pulses(ax, rf, pulses='pulse*.oct.dat'):
    for i, pulse_file in enumerate(sorted(glob(join(rf, pulses)))):
        p = QDYN.pulse.Pulse.read(pulse_file)
        p.render_pulse(ax, label='pulse %d' % (i+1))
    ax.legend()


def render_population(ax, rf, single_node=False):
    qubit_pop = np.genfromtxt(join(rf, 'qubit_pop.dat')).transpose()
    tgrid = qubit_pop[0]
    if single_node:
        ax.plot(tgrid, qubit_pop[1], label=r'0')
        ax.plot(tgrid, qubit_pop[2], label=r'1')
        total = qubit_pop[1] + qubit_pop[2]
    else:
        ax.plot(tgrid, qubit_pop[1], label=r'00')
        ax.plot(tgrid, qubit_pop[2], label=r'01')
        ax.plot(tgrid, qubit_pop[3], label=r'10')
        ax.plot(tgrid, qubit_pop[4], label=r'11')
        total = qubit_pop[1] + qubit_pop[2] + qubit_pop[3] + qubit_pop[4]
    ax.plot(tgrid, total, label=r'total', ls='--')
    ax.legend(loc='best', fancybox=True, framealpha=0.5)
    ax.set_ylim([0, 1.1])
    ax.set_xlabel("time")
    ax.set_ylabel("population")


def render_excitation(ax, rf, filter=''):
    exc_file = join(rf, 'excitation.dat')
    with open(exc_file) as in_fh:
        header = in_fh.readline()
        labels = header.strip('#').strip().split()[2:]
    excitation = np.genfromtxt(exc_file).transpose()
    tgrid = excitation[0]
    total = np.zeros(len(tgrid))
    accept = lambda label: True
    render_label = lambda label: label
    accept = lambda label: label.startswith(filter)
    render_label = lambda label: label[1:]
    for i, label in enumerate(labels):
        if accept(label):
            total += excitation[i+1]
            ax.plot(tgrid, excitation[i+1], label=render_label(label))
    ax.plot(tgrid, total, ls='--', label='total')
    ax.legend(loc='best', fancybox=True, framealpha=0.5)
    ax.set_ylim([0, max(1.0, 1.05*ax.get_ylim()[1])])
    ax.set_xlabel("time")
    if len(filter) > 0:
        ax.set_ylabel("%s excitation" % filter)
    else:
        ax.set_ylabel("excitation")


def display_with_cc(expr):
    from qnet.algebra.abstract_algebra import extra_binary_rules
    from qnet.algebra.operator_algebra import OperatorPlus, create_operator_pm_cc
    with extra_binary_rules(OperatorPlus, create_operator_pm_cc()):
        display(expr.simplify())


def display_eq(lhs, rhs):
    """Display the eqution defined by the given lhs and rhs"""
    lines = []
    lines.append(r'\begin{equation}')
    lines.append(tex(lhs) + ' = ' + tex(rhs))
    lines.append(r'\end{equation}')
    display(Latex("\n".join(lines)))


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
                ind = s.split("_")[-1]
            except ValueError:
                print(s)
                raise
            if len(ind) > 1:
                ind = r'\%s' % ind
            return r'\hat{H}_{%s}' % ind

    lines = []
    lines.append(r'\begin{align}')
    lines.append(r'  \hat{H} &= %s\\' % " + ".join(
        [label(name) for name in terms.keys()]))
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
        ('t', tgrid),
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


def plot_bs_decay(L):
    """Under the assumption that `L` is a sum of operator with identical
    prefactors, plot that common prefactor in dependence of the symbol theta,
    in units of the symbol sqrt(2*kappa)
    """
    θ = Symbol('theta', real=True)
    κ = Symbol('kappa', positive=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    theta = np.linspace(0, 2*np.pi, 100)
    cs = [op.coeff for op in pattern(ScalarTimesOperator).findall(L)]
    for c in cs[1:]:
        assert c == cs[0]
    c = cs[0] / sympy.sqrt(2*κ)
    k = np.array([c.subs({θ: val}) for val in theta])
    ax.plot(theta/np.pi, k)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r'BS mixing angle $\theta$ ($\pi$ rad)')
    ax.set_ylabel(r'decay rate (\sqrt(2\kappa)')
    plt.show(fig)


def monotonic_convergence(iter, J_T):
    iter_out = []
    J_T_out = []
    J_T_prev = None
    for iter_val, J_T_val in zip(iter, J_T):
        if J_T_prev is not None:
            if J_T_prev < J_T_val:
                continue  # drop
        iter_out.append(iter_val)
        J_T_out.append(J_T_val)
        J_T_prev = J_T_val
    return iter_out, J_T_out


def plot_convergence_comparison(runfolders, monotonic=False, xlim=None):
    conv = QDYN.octutils.OCTConvergences()
    for rf in runfolders:
        try:
            conv.load_file(
                rf.split('/')[-1],
                join(rf, 'oct_iters.dat'))
        except FileNotFoundError:
            pass
    fig, ax = plt.subplots(figsize=(10, 6))
    for (key, data) in conv.data.items():
        if monotonic:
            ax.plot(*monotonic_convergence(data.iter, data.J_T), label=key)
        else:
            ax.plot(data.iter, data.J_T, label=key)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel("iteration")
    ax.set_ylabel("error")
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.show(fig)


def plot_rho_prop_error_comparison(runfolders, xlim=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for rf in runfolders:
        key = rf.split('/')[-1]
        oct_iter, err = np.genfromtxt(
            join(rf, 'rho_prop_error.dat'), unpack=True)
        ax.plot(oct_iter, err, label=key)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel("iteration")
    ax.set_ylabel("error")
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.show(fig)


def plot_pulse_comparison(runfolders):
    ncols = 3
    nrows = len(runfolders) // ncols
    if ncols * nrows < len(runfolders):
        nrows += 1
    fig, axs = plt.subplots(
        figsize=(16, nrows*5), ncols=ncols, nrows=nrows, squeeze=False)
    axs = axs.flatten()
    for i, rf in enumerate(runfolders):
        ax = axs[i]
        pulse1 = QDYN.pulse.Pulse.read(join(rf, 'pulse1.oct.dat'))
        pulse2 = QDYN.pulse.Pulse.read(join(rf, 'pulse2.oct.dat'))
        pulse1.render_pulse(ax, label=r'pulse 1')
        pulse2.render_pulse(ax, label=r'pulse 2')
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        ax.set_title(rf.split('/')[-1])
    plt.show(fig)


def render_pulse_delta_smoothing(
        ax, rf, smooth_pulse, pulse_ids=(1, 2), labels=None, **kwargs):
    if labels is not None:
        labels = list(labels)
    pulses = [
        QDYN.pulse.Pulse.read(join(rf, 'pulse%d.oct.dat' % id))
        for id in pulse_ids]
    delta_pulses = pulse_delta_smoothing(pulses, smooth_pulse, **kwargs)
    for id, delta_pulse in zip(pulse_ids, delta_pulses):
        if labels is None:
            label = r'Δ' + SUBSCRIPT_MAPPING[str(id)]
        else:
            label = labels.pop(0)
        delta_pulse.render_pulse(ax, label=label)
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("noise amplitude")
    ax.set_title(rf.split('/')[-1])


def plot_pulse_delta_smoothing(runfolders, smooth_pulse, **kwargs):
    """Plot pulses relative to smooth version of itself"""
    ncols = 3
    nrows = len(runfolders) // ncols
    if ncols * nrows < len(runfolders):
        nrows += 1
    fig, axs = plt.subplots(
        figsize=(16, nrows*5), ncols=ncols, nrows=nrows, squeeze=False)
    axs = axs.flatten()
    for i, rf in enumerate(runfolders):
        ax = axs[i]
        render_pulse_delta_smoothing(ax, rf, smooth_pulse, **kwargs)
    plt.show(fig)


def collect_noise_table(runfolders, smooth_pulse, **kwargs):
    noise1 = []  # pulse 1
    noise2 = []  # pulse 2
    ntrajs = []
    for rf in runfolders:
        ntrajs.append(int(rf.split('ntrajs')[-1]))
        pulse1 = QDYN.pulse.Pulse.read(join(rf, 'pulse1.oct.dat'))
        pulse2 = QDYN.pulse.Pulse.read(join(rf, 'pulse2.oct.dat'))
        delta1, delta2 = pulse_delta_smoothing(
            [pulse1, pulse2], smooth_pulse, **kwargs)
        dt = pulse1.tgrid[1] - pulse1.tgrid[0]
        pulse1_noise = sum(np.abs(delta1.amplitude)) * dt
        pulse2_noise = sum(np.abs(delta2.amplitude)) * dt
        noise1.append(pulse1_noise)
        noise2.append(pulse2_noise)
    return pd.DataFrame(
        data=OrderedDict(
            [('ν1', noise1), ('ν2', noise2), ('rf', runfolders)]),
        index=ntrajs)


def combine_noise_tables(noise_tables, labels):
    return (
        pd.concat(
            [df[['ν1', 'ν2']].rename(
                columns={'ν1': 'ν1 ' + label, 'ν2': 'ν2 ' + label})
             for (df, label) in zip(noise_tables, labels)],
            axis=1)
        .dropna())
