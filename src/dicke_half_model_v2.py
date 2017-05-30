"""Routines for constructing guess pulses"""
from os.path import join
from glob import glob

import sympy
import qutip
from sympy import Symbol
import numpy as np

from qnet.algebra import ScalarTimesOperator, pattern

import QDYN
from QDYN.pulse import blackman
from QDYN.units import UnitFloat

from .qdyn_model_v1 import make_qdyn_oct_model, dicke_state, err_state_to_state


def pi_pulse(tgrid, t_start, t_stop, mu, cycles=1):
    """Return a blackman Pulse between t_start and t_stop with an amplitude
    that produces a pi-pulse (population inversion) in a two-level system with
    coupling strength mu

    Assumes dimensionless units
    """
    from numpy import pi
    E0 = pi / (mu * (t_stop - t_start) * (1.0 - 0.16))
    pulse = QDYN.pulse.Pulse(
        tgrid, amplitude=(cycles * E0 * blackman(tgrid, t_start, t_stop)),
        time_unit='dimensionless', ampl_unit='dimensionless',
        freq_unit='dimensionless')
    assert not pulse.is_complex
    return pulse


def zero_pulse(tgrid):
    """Return a Pulse with zero amplitude, with dimensionless units"""
    pulse = QDYN.pulse.Pulse(
        tgrid, amplitude=(0.0 * tgrid),
        time_unit='dimensionless', ampl_unit='dimensionless',
        freq_unit='dimensionless')
    assert not pulse.is_complex
    return pulse


def num_vals(theta=0.0, n_nodes=2, kappa=0.01):
    """Fixed numerical parameters"""
    Delta = 100.0
    g = 1.0  # by definition
    theta = theta * np.pi
    num_vals = {
        Symbol('kappa', positive=True): kappa,
        Symbol('theta', real=True): theta,
        Symbol('lambda', real=True): g / Delta,
        Symbol('Omega_alpha'): 0.0
    }
    for i_node in range(n_nodes):
        num_vals[Symbol('Delta_%d' % (i_node + 1), real=True)] = Delta
        num_vals[Symbol('g_%d' % (i_node + 1), positive=True)] = g
    return num_vals


def get_mu_1(slh, num_vals):
    ctrl_sym = sympy.Symbol(r'Omega_1')
    coeffs = [t.coeff for t in pattern(ScalarTimesOperator).findall(slh.H)
              if ctrl_sym in t.all_symbols()]
    return (abs(coeffs[0]) / abs(ctrl_sym)).subs(num_vals)


def dicke_guess_controls(slh, theta, T, E0_cycles=2, nt=None, kappa=0.01):
    tgrid_start = 0
    tgrid_end = T
    n_nodes = slh.n_nodes
    if nt is None:
        nt = int(tgrid_end - tgrid_start) + 1
    tgrid = QDYN.pulse.pulse_tgrid(t0=tgrid_start, T=tgrid_end, nt=nt)
    mu_1 = float(get_mu_1(slh, num_vals(theta=theta, n_nodes=n_nodes,
                                        kappa=kappa)))
    controls = {}
    dt = tgrid[1] - tgrid[0]
    t_rise = UnitFloat(max(0.01 * T, 4*dt), 'dimensionless')
    pulse_sideband = pi_pulse(tgrid, t_start=tgrid_start, t_stop=tgrid_end,
                              mu=mu_1, cycles=E0_cycles)
    E0 = UnitFloat(
        np.max(np.abs(pulse_sideband.amplitude)),
        pulse_sideband.ampl_unit)
    pulse_sideband.config_attribs['E_0'] = E0
    pulse_sideband.config_attribs['t_rise'] = t_rise
    pulse_sideband.config_attribs['t_fall'] = t_rise
    for ctrl_sym in [Symbol('Omega_%d' % ind) for ind in range(1, n_nodes+1)]:
        controls[ctrl_sym] = pulse_sideband.copy()
    return controls


def write_dicke_half_model(
        slh, rf, T, theta=0, E0_cycles=2, mcwf=False, non_herm=False,
        lambda_a=1.0, J_T_conv=1e-4, iter_stop=5000, nt=None,
        max_ram_mb=100000, kappa=0.01, complex_pulses=False):
    """Write model to the given runfolder,

    Args:
        slh: the symbolic SLH model of the circuit to simulate. Must have
            attribute n_nodes.
        rf (str): the path of the runfolder to write
        T (int): The duration of the entire process, in dimensionless units.
            Could be a float, but round numbers (ints) are appreciated
        theta (float): The mixing angle of the beamsplitter in the circuit. A
            value of 0.0 means full transmission (the same as no beamsplitter)
        E0_cycles: The amplitude of all the guess pulses, as the number of
            pi-cycles (population inversions in a two-level systems)
        mcwf: Whether or not to use the MCWF method
        mcwf (bool): If True, initialize for MCWF propagation. Otherwise,
            non-dissipative Hilbert space or Liouville space propagation. See
            below for details.
        non_herm (bool): If True, add the decay term to the Hamiltonian. If
            used together with `mcwf=True`, this improves efficiency slightly.
            If used with `mcwf=False`, a non-Hermitian Hamiltonian will be
            propagated. See below for details
        lambda_a (float): The value of the lambda_a Krotov scaling parameter
        J_T_conv (float): Value of functional we want to reach
        iter_stop (int): Max number of OCT iterations
        nt (int): Number of time step (1 per time unit if None)
        max_ram_mb (int): MB of RAM to use for storing propagated states
        kappa (float): local decay rate of each node
        complex_puses (bool): If True, allow for complex pulse shapes

    Note:

        The "propagation mode" is determined by the values of `mcwf` and
        `non_herm`. The following combinations are possible:

        * density matrix propagation: ``mcwf=False``, ``non_herm=False``
        * non-Hermitian propagation in Hilbert space:
          ``mcwf=False``, ``non_herm=True``
        * MCWF propagation with effective Hamiltonian auto-calculated from
          Lindblad operators in QDYN:
          ``mcwf=True, ``non_herm=False``
        * MCWF propagation with pre-computed effective Hamiltonian:
          ``mcwf=True, ``non_herm=True``
    """
    n_nodes = slh.n_nodes
    controls = dicke_guess_controls(slh=slh, theta=theta, T=T,
                                    E0_cycles=E0_cycles, nt=nt, kappa=kappa)
    for pulse in controls.values():
        assert not pulse.is_complex
        if complex_pulses:
            pulse.amplitude = np.array(pulse.amplitude, dtype=np.complex128)
            assert pulse.is_complex

    qdyn_model = make_qdyn_oct_model(
        slh, num_vals=num_vals(theta=theta, n_nodes=n_nodes, kappa=kappa),
        controls=controls, energy_unit='dimensionless',
        mcwf=mcwf, non_herm=non_herm, oct_target='dicke_half',
        lambda_a=lambda_a, iter_stop=iter_stop, keep_pulses='prev',
        J_T_conv=J_T_conv, max_ram_mb=max_ram_mb)
    qdyn_model.user_data['state_label'] = 'dicke_init_half'  # for prop
    if (mcwf, non_herm) == (False, False):
        qdyn_model.user_data['rho'] = True
    qdyn_model.write_to_runfolder(rf)


def err_dicke_half(rf, final_states_glob, rho=False):
    config = QDYN.config.read_config_file(join(rf, 'config'))
    n = int(config['ham'][0]['n_surf'])
    psi_dicke_half = QDYN.state.read_psi_amplitudes(
            join(rf, "psi_dicke_half.dat"), n)
    if rho:
        psi_dicke_half = qutip.Qobj(psi_dicke_half)
        rho_dicke_half = psi_dicke_half * psi_dicke_half.dag()
        final_state_files = glob(join(rf, final_states_glob))
        assert len(final_state_files) == 1
        final_rho_fname = final_state_files[0]
        final_rho = qutip.Qobj(QDYN.io.read_indexed_matrix(
            final_rho_fname, shape=(n, n)))
        err = 1.0 - (rho_dicke_half * final_rho).tr()
        assert err.imag < 1e-14
        return err.real
    else:
        return err_state_to_state(psi_dicke_half, join(rf, final_states_glob))
