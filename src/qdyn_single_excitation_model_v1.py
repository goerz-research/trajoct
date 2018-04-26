"""Routines to construct the QDYN model for a multi-node network with linear
controls, using the zero/single-excitation subspace encoding"""

from collections import OrderedDict

import qutip
from scipy import sparse
import QDYN
import numpy as np
from QDYN.units import UnitFloat
from QDYN.dissipation import lindblad_ops_to_dissipator
from QDYN.linalg import triu, tril

from qnet.algebra.operator_algebra import (
    Destroy, Create, LocalSigma, ZeroOperator)
from qnet.algebra.pattern_matching import pattern, wc

from .algebra_v1 import split_hamiltonian
from .to_single_excitation_qutip import (
    convert_to_single_excitation_qutip, construct_bit_index, _qutip_sigma)
from .qdyn_model_v2 import H0_eff, pulses_uptodate, prepare_rf_prop


__all__ = [
    'make_single_excitation_qdyn_model',
    'make_single_excitation_qdyn_oct_model',
    'pulses_uptodate', 'prepare_rf_prop',
    'single_excitation_dicke_state', 'single_excitation_initial_state'
    ]


def _state_to_fmt(qutip_state, fmt):
    import QDYN.linalg
    if fmt == 'qutip':
        return qutip_state
    elif fmt == 'numpy':
        return QDYN.linalg.vectorize(qutip_state.data.todense())
    else:
        raise ValueError("Unknown fmt")


def single_excitation_dicke_state(bit_index, fmt='qutip'):
    """Return the zero/single-excitation dicke state for the given system,
    directly using the single-exctiation subspace encoding. Assumes
    that the Hilbert spaces for the qubits at the different nodes are labeled
    'q1', 'q2', ...

    For example for a Hilbert space ``('q1', 'c1', 'q2', 'c2', 'q3', 'c3')``,
    the Dicke state is ``(|000010> + |001000> + |100000>) / sqrt(3)``. The
    corresponding bit-index is

        {'000000': 0,
         '000001': 1,
         '000010': 2,
         '000100': 3,
         '001000': 4,
         '010000': 5,
         '100000': 6}

    and only the eigenstates 2, 4, and 6 contribute to the Dicke state.
    """
    N = len(bit_index)
    terms = [qutip.basis(N, i) for i in range(2, N, 2)]
    res = sum(terms) / np.sqrt(len(terms))
    return _state_to_fmt(res, fmt)


def single_excitation_initial_state(bit_index, fmt='qutip'):
    """Return the initial state "e0....0" directly using the single-excitation
    subspace enconding"""
    N = len(bit_index)
    i = bit_index["1" + "0" * (N-2)]
    assert i == N - 1
    return _state_to_fmt(qutip.basis(N, i), fmt)


def make_single_excitation_qdyn_model(
        network_slh, num_vals, controls, energy_unit='MHz',
        mcwf=False, non_herm=False, use_dissipation_superop=True,
        states=None, nodiss=False, observables='all'):
    """Construct a QDYN LevelModel for a network of one or more nodes,
    configured for propagation

    Args:
        network_slh (SLH): SLH model of the system
        num_vals (dict): mapping of symbols to numerical values
        controls (OrderedDict): mapping of control symbols to Pulse instances.
        energy_unit (str): The physical unit of the drift Hamiltonian. The
            Lindbladians must be in units of sqrt-`energy_unit`.
        mcwf (bool): If True, initialize for MCWF propagation. Otherwise,
            non-dissipative Hilbert space or Liouville space propagation. See
            below for details.
        non_herm (bool): If True, add the decay term to the Hamiltonian. If
            used together with `mcwf=True`, this improves efficiency slightly.
            If used with `mcwf=False`, a non-Hermitian Hamiltonian will be
            propagated. See below for details
        use_dissipation_superop (bool): If True, pre-compute a dissipation
            superoperator instead of using Lindblad Operators. This only
            affects density matric propagations (``mcwf=False``,
            ``non_herm=False``
        states (dict or None): dict label => state (numpy array of amplitudes).
            If None, no state will be added to the model.
        nodiss (bool): If True, drop any dissipation from the system
        observables (str): One of 'all', 'last'.

    Notes:

        The "propagation mode" is determined by the values of `mcwf` and
        `non_herm`. The following combinations are possible:

        * density matrix propagation: ``mcwf=False``, ``non_herm=False``
        * non-Hermitian propagation in Hilbert space:
          ``mcwf=False``, ``non_herm=True``
        * MCWF propagation with effective Hamiltonian auto-calculated from
          Lindblad operators in QDYN:
          ``mcwf=True, ``non_herm=False``
        * MCWF propagation with pre-computed effective Hamiltonian

        The propagation time grid is automatically taken from the control
        pulses.

        The control Hamiltonians must be dimensionless. Thus, the controls
        are in an energy unit specified in the `energy_unit` attribute of the
        corresponding Pulse instance which may or may not be the same as the
        `energy_unit` argument.

        The resulting model defines no OCT parameters. If desired, these must
        be added separately.
    """

    control_syms = list(controls.keys())

    H_num = network_slh.H.substitute(num_vals, fast=True)
    ham_parts = split_hamiltonian(
        H_num, use_cc=False, controls=control_syms, fast=True)
    hs = H_num.space

    bit_index = construct_bit_index(hs)

    # time grid (determined by pulses)
    pulses = list(controls.values())
    nt = len(pulses[0].tgrid) + 1
    t0 = pulses[0].t0
    T = pulses[0].T
    time_unit = pulses[0].time_unit
    for pulse in pulses[1:]:
        assert len(pulse.tgrid) + 1 == nt
        assert pulse.t0 == t0
        assert pulse.T == T
        assert pulse.time_unit == time_unit

    hs_labels = [ls.label for ls in hs.local_factors]
    assert 'q1' in hs_labels
    n_nodes = len([l for l in hs_labels if l.startswith('q')])

    pat_localsigma = pattern(LocalSigma, wc('i'), wc('j'), hs=wc('hs'))

    def localsigma_to_create_destroy(i, j, hs):
        if (i, j) == ('g', 'e'):
            return Destroy(hs=hs)
        elif (i, j) == ('e', 'g'):
            return Create(hs=hs)
        else:
            raise ValueError("Invalid (i, j) = %s" % (str((i, j))))

    for (key, H) in ham_parts.items():
        ham_parts[key] = H.simplify(
            [(pat_localsigma, localsigma_to_create_destroy)])

    if ham_parts['Hint'] == ZeroOperator:
        # split_hamiltonian with 'fast=True' does not separate H0 and Hint
        H0 = convert_to_single_excitation_qutip(
            ham_parts['H0'], bit_index, full_space=hs)
    else:
        H0 = (
            convert_to_single_excitation_qutip(
                ham_parts['H0'], bit_index, full_space=hs) +
            convert_to_single_excitation_qutip(
                ham_parts['Hint'], bit_index, full_space=hs))

    if nodiss:
        Ls = []
    else:
        Ls = [
            convert_to_single_excitation_qutip(
                L.substitute(num_vals, fast=True), bit_index, full_space=hs)
            for L in network_slh.Ls]

    model = QDYN.model.LevelModel()

    # depending on propagation mode, set appropriate drift Hamiltonian,
    # dissipation and prop settings

    if (mcwf, non_herm) == (False, False):

        # Density matrix propagation
        model.add_ham(H0, op_unit=energy_unit, op_type='potential')
        if use_dissipation_superop:
            D = lindblad_ops_to_dissipator(
                [sparse.coo_matrix(L.data) for L in Ls])
            model.set_dissipator(D, op_unit=energy_unit)
        else:
            for L in Ls:
                lindblad_unit = 'sqrt_%s' % energy_unit
                if energy_unit in ['dimensionless', 'unitless', 'iu']:
                    lindblad_unit = energy_unit
                model.add_lindblad_op(L, op_unit=lindblad_unit)
        model.set_propagation(
            T=T, nt=nt, t0=t0, time_unit=time_unit, prop_method='newton')

    elif (mcwf, non_herm) == (False, True):

        # Propagation in Hilbert space with a non-Hermitian Hamiltonian
        model.add_ham(H0_eff(H0, Ls), op_unit=energy_unit, op_type='potential')
        model.set_propagation(
            T=T, nt=nt, t0=t0, time_unit=time_unit, prop_method='newton')

    elif mcwf:

        # MCWF propagation
        for L in Ls:
            lindblad_unit = 'sqrt_%s' % energy_unit
            if energy_unit in ['dimensionless', 'unitless', 'iu']:
                lindblad_unit = energy_unit
            if non_herm:
                # We precompute the effective Hamiltonian
                model.add_lindblad_op(
                    L, op_unit=lindblad_unit, conv_to_superop=False)
            else:
                # QDYN determines the effective Hamiltonian internally
                model.add_lindblad_op(
                    L, op_unit=lindblad_unit, add_to_H_jump='indexed',
                    conv_to_superop=False)
        if non_herm:
            model.add_ham(H0_eff(H0, Ls), op_unit=energy_unit,
                          op_type='potential')
            model.set_propagation(
                T=T, nt=nt, t0=t0, time_unit=time_unit, prop_method='newton',
                use_mcwf=True, mcwf_order=2, construct_mcwf_ham=False)
        else:
            model.add_ham(H0, op_unit=energy_unit, op_type='potential')
            model.set_propagation(
                T=T, nt=nt, t0=t0, time_unit=time_unit, prop_method='newton',
                use_mcwf=True, mcwf_order=2, construct_mcwf_ham=True)

    # control Hamiltonians
    for i, control_sym in enumerate(control_syms):
        H_d = (
            ham_parts['H_%s' % (str(control_sym))]
            .substitute({control_sym: 1})
            .simplify([(pat_localsigma, localsigma_to_create_destroy)]))
        H_d = convert_to_single_excitation_qutip(H_d, bit_index, full_space=hs)
        pulse = controls[control_sym]
        if pulse.is_complex:
            H_d_u = qutip.Qobj(triu(H_d))
            H_d_l = qutip.Qobj(tril(H_d))
            model.add_ham(H_d_u, pulse=pulse, op_unit='dimensionless',
                          op_type='dipole')
            model.add_ham(H_d_l, pulse=pulse, op_unit='dimensionless',
                          op_type='dipole', conjg_pulse=True)
        else:
            model.add_ham(H_d, pulse=pulse, op_unit='dimensionless',
                          op_type='dipole')

    # states
    if states is not None:
        for label, psi in states.items():
            assert psi.shape[0] == H0.shape[0]
            model.add_state(psi, label)

    # observables

    if observables == 'all':
        from_time_index, to_time_index, step = 1, -1, 1
    elif observables == 'last':
        from_time_index, to_time_index, step = -1, -1, 1
    else:
        raise ValueError("Invalid 'observables': %s" % observables)

    time_unit = 'microsec'
    if energy_unit == 'dimensionless':
        time_unit = 'dimensionless'
    if not nodiss:
        L_total = Ls[0].dag() * Ls[0]
        for L in Ls[1:]:
            L_total += L.dag() * L
        if L_total.norm('max') > 1e-14:
            model.add_observable(
                L_total, outfile='darkstate_cond.dat', exp_unit=energy_unit,
                time_unit=time_unit, col_label='<L^+L>', is_real=True,
                from_time_index=from_time_index, to_time_index=to_time_index,
                step=step)
    assert n_nodes >= 2
    for ls in hs.local_factors:
        n_op = _qutip_sigma(ls, ls, hs, bit_index)
        model.add_observable(
            n_op, outfile='excitation.dat', exp_unit='dimensionless',
            time_unit=time_unit, col_label=ls.label, is_real=True,
            from_time_index=from_time_index, to_time_index=to_time_index,
            step=step)

    model.user_data['time_unit'] = time_unit
    model.user_data['write_jump_record'] = 'jump_record.dat'
    model.user_data['write_final_state'] = 'psi_final.dat'
    return model


def make_single_excitation_qdyn_oct_model(
        network_slh, num_vals, controls, *, oct_target, energy_unit='MHz',
        mcwf=False, non_herm=False, use_dissipation_superop=True,
        lambda_a=1e-5, seed=None, allow_negative=True, observables='all',
        **kwargs):
    """Construct a QDYN level model for OCT, for two or more nodes"""
    import logging
    logger = logging.getLogger(__name__)
    hs = network_slh.H.space
    bit_index = construct_bit_index(hs)
    labels = [ls.label for ls in hs.local_factors]
    n_nodes = len([l for l in labels if l.startswith('q')])
    assert 1 < n_nodes < 200
    psi10 = single_excitation_initial_state(bit_index)
    dicke_1 = single_excitation_dicke_state(bit_index)
    states = OrderedDict([('10', psi10), ('dicke_1', dicke_1)])

    model = make_single_excitation_qdyn_model(
        network_slh, num_vals, controls, energy_unit=energy_unit, mcwf=mcwf,
        non_herm=non_herm, use_dissipation_superop=use_dissipation_superop,
        states=states, observables=observables)

    for i, pulse in enumerate(model.pulses()):

        E_max = 10 * UnitFloat(np.max(np.abs(pulse.amplitude)),
                               pulse.ampl_unit)
        if allow_negative:
            E_min = - E_max
        else:
            E_min = 0

        def set_pulse_setting(pulse, key, val, force=False):
            """Set pulse.config_attribs[key] = val unless that key is already
            set"""
            if force and pulse.config_attribs.get(key, val) != val:
                logger.warning("Overwriting pulse parameter %s: %s -> %s"
                               % (key, pulse.config_attribs[key], val))
                pulse.config_attribs[key] = val
            else:
                pulse.config_attribs[key] = pulse.config_attribs.get(key, val)

        set_pulse_setting(pulse, 'oct_shape', 'flattop')
        set_pulse_setting(pulse, 'shape_t_start', pulse.t0)
        set_pulse_setting(pulse, 'shape_t_stop', pulse.T)
        set_pulse_setting(pulse, 't_rise', 0.1*(pulse.T-pulse.t0))
        set_pulse_setting(pulse, 't_fall', 0.1*(pulse.T-pulse.t0))
        set_pulse_setting(pulse, 'oct_lambda_a', lambda_a, force=True)
        set_pulse_setting(pulse, 'oct_outfile', 'pulse%d.oct.dat' % (i+1))
        set_pulse_setting(pulse, 'oct_increase_factor', 5)
        set_pulse_setting(pulse, 'oct_pulse_max', E_max)
        set_pulse_setting(pulse, 'oct_pulse_min',  E_min)

    oct_settings = OrderedDict([
        ('method', 'krotovpk'),
        ('J_T_conv', 1e-4),
        ('max_ram_mb', 100000),
        ('iter_dat', 'oct_iters.dat'),
        ('iter_stop', 100),
        ('tau_dat', 'oct_tau.dat'),
        ('params_file', 'oct_params.dat'),
        ('limit_pulses', True),
        ('keep_pulses', 'all'),
        ])
    oct_settings.update(kwargs)

    model.set_oct(**oct_settings)

    if oct_target == 'dicke_1':
        model.user_data['initial_states'] = '10'
        model.user_data['target_states'] = 'dicke_1'
        P_target = dicke_1 * dicke_1.dag()
    else:
        raise ValueError("Unknown oct_target %s" % oct_target)

    # observables
    if observables == 'all':
        from_time_index, to_time_index, step = 1, -1, 1
    elif observables == 'last':
        from_time_index, to_time_index, step = -1, -1, 1
    else:
        raise ValueError("Invalid 'observables': %s" % observables)

    if oct_target != 'gate':
        model.add_observable(
            P_target, outfile='P_target.dat', exp_unit='dimensionless',
            time_unit='dimensionless', col_label='<P>', square='<P^2>',
            from_time_index=from_time_index, to_time_index=to_time_index,
            step=step)

    if seed is not None:
        model.user_data['seed'] = seed

    return model
