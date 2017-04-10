"""Routines to construct the QDYN model for a multi-node network with linear
controls"""

from os.path import join
from glob import glob
from collections import OrderedDict
import shutil

import qutip
from scipy import sparse
import QDYN
import numpy as np
from QDYN.units import UnitFloat
from QDYN.dissipation import lindblad_ops_to_dissipator
from QDYN.linalg import triu, tril

from qnet.algebra.operator_algebra import Destroy, Create
from qnet.algebra.circuit_algebra import move_drive_to_H
from qnet.convert.to_qutip import convert_to_qutip

from algebra import split_hamiltonian


def dagger(op):
    return op.adjoint()


def state(space, *numbers, fmt='qutip'):
    """Construct a state for a given QNET space by giving a quantum number for
    each sub-space

    Args:
    space (qnet.algebra.hilbert_space_algebra.HilbertSpace): The space in which
        the state lives
    numbers (tuple of ints): 0-based quantum numbers, one for each local factor
        in `space`
    fmt (str): output format. 'qutip' for a QuTiP state, 'numpy' for a numpy
        complex vector
    """
    states = []
    assert len(numbers) == len(space.local_factors)
    for i, hs in enumerate(space.local_factors):
        states.append(qutip.basis(hs.dimension, numbers[i]))
    if fmt == 'qutip':
        return qutip.tensor(*states)
    elif fmt == 'numpy':
        return QDYN.linalg.vectorize(qutip.tensor(*states).data.todense())
    else:
        raise ValueError("Unknown fmt")


def logical_2q_state(space, i, j, fmt='qutip'):
    """Return logical two-qubit state |ij>, defined accross the first and last
    node, under the (checked) assumption that the qubit Hilbert spaces are
    labeled as "q<i>" where <i> is the (1-based) index of the nodes
    """
    assert i in [0, 1] and j in [0, 1]
    qnums = []
    found_q1 = False
    found_qn = False
    n_nodes = len([ls.label for ls in space.local_factors
                   if ls.label.startswith('q')])
    qn_label = 'q%d' % n_nodes
    for ls in space.local_factors:
        if ls.label == 'q1':
            qnums.append(i)
            found_q1 = True
        elif ls.label == qn_label:
            qnums.append(j)
            found_qn = True
        else:
            qnums.append(0)
    assert found_q1 and found_qn
    return state(space, *qnums, fmt=fmt)


def logical_1q_state(space, i, fmt='qutip'):
    """Return logical single-qubit state |i> on space 'q1'"""
    assert i in [0, 1]
    qnums = []
    found_q1 = False
    for ls in space.local_factors:
        if ls.label == 'q1':
            qnums.append(i)
            found_q1 = True
        else:
            qnums.append(0)
    assert found_q1
    return state(space, *qnums, fmt=fmt)


def dicke_state(space, fmt='qutip', excitations=1):
    """Return the single-excitation dicke state for the given system. Assumes
    that the Hilbert spaces for the qubits at the different nodes are labeled
    'q1', 'q2', ...

    For example for a Hilbert space ``('q1', 'c1', 'q2', 'c2', 'q3', 'c3')``,
    the Dicke state is ``(|000010> + |001000> + |100000>) / sqrt(3)``
    """
    assert excitations == 1
    res = None
    labels = [ls.label for ls in space.local_factors]
    n_nodes = len([l for l in labels if l.startswith('q')])
    for i_node in range(n_nodes):
        label = 'q%d' % (i_node + 1)
        qnums = [1 if l == label else 0 for l in labels]
        if res is None:
            res = state(space, *qnums, fmt=fmt)
        else:
            res += state(space, *qnums, fmt=fmt)
    return res / np.sqrt(n_nodes)


def dicke_init_state(space, fmt='qutip', excitations=1):
    """Return the the state of the first $n$ nodes are in the excited states,
    wheren $n$ is given by `excitations`
    """
    labels = [ls.label for ls in space.local_factors]
    qnums = []
    for label in labels:
        if label.startswith('q') and excitations > 0:
            qnums.append(1)
            excitations = excitations - 1
        else:
            qnums.append(0)
    return state(space, *qnums, fmt=fmt)


def err_state_to_state(target_state, final_states_glob):
    """Return error of a single state-to-state transfer

    Args:
        target_state (array): numpy array of amplitudes of target state
        final_states_glob (str): string that expands to a list of files
            from which propagated states should be read (for multiple
            trajectories)
    """
    final_state_files = glob(final_states_glob)
    target_state = QDYN.linalg.vectorize(target_state)
    n = len(final_state_files)
    assert n > 0
    F = 0.0
    n_hilbert = len(target_state)
    for fn in final_state_files:
        final_state = QDYN.state.read_psi_amplitudes(fn, n=n_hilbert)
        F += abs(QDYN.linalg.inner(target_state, final_state))**2
    return 1.0 - F/float(n)


def H0_eff(H0, Ls):
    """Return a modified H0 that is the effective non-Hermitian Hamiltonian in
    an MCWF propagation (or just a non-Hermitian Hilbert space propagation)
    """
    for L in Ls:
        H0 = H0 - 0.5j*(L.dag() * L)
    return H0


def make_qdyn_model(
        network_slh, num_vals, controls, energy_unit='MHz',
        mcwf=False, non_herm=False, use_dissipation_superop=True,
        states=None, nodiss=False):
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

    network_slh = move_drive_to_H(network_slh)

    H_num = network_slh.H.substitute(num_vals)
    ham_parts = split_hamiltonian(H_num, use_cc=False, controls=control_syms)
    hs = H_num.space

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

    H0 = (convert_to_qutip(ham_parts['H0'], full_space=hs) +
          convert_to_qutip(ham_parts['Hint'], full_space=hs))

    Ls = [convert_to_qutip(L.substitute(num_vals), full_space=hs)
          for L in network_slh.Ls]
    if nodiss:
        Ls = []

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
            model.add_lindblad_op(L, op_unit=lindblad_unit,
                                  add_to_H_jump='indexed',
                                  conv_to_superop=False)
        if non_herm:
            # We precompute the effective Hamiltonian
            model.add_ham(H0_eff(H0, Ls), op_unit=energy_unit,
                          op_type='potential')
            model.set_propagation(
                T=T, nt=nt, t0=t0, time_unit=time_unit, prop_method='newton',
                use_mcwf=True, mcwf_order=2, construct_mcwf_ham=False)
        else:
            # QDYN determines the effective Hamiltonian internally
            model.add_ham(H0, op_unit=energy_unit, op_type='potential')
            model.set_propagation(
                T=T, nt=nt, t0=t0, time_unit=time_unit, prop_method='newton',
                use_mcwf=True, mcwf_order=2, construct_mcwf_ham=True)

    # control Hamiltonians
    for i, control_sym in enumerate(control_syms):
        H_d = convert_to_qutip(
            ham_parts['H_%s' % (str(control_sym))].substitute(
                {control_sym: 1}),
            full_space=hs)
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
                time_unit=time_unit, col_label='<L^+L>', is_real=True)
    if n_nodes >= 2:
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            ket = logical_2q_state(hs, i, j)
            rho = ket * ket.dag()
            model.add_observable(
                rho, outfile='qubit_pop.dat', exp_unit='dimensionless',
                time_unit=time_unit, col_label='P(%d%d)' % (i, j),
                is_real=True)
        rho = logical_2q_state(hs, 1, 0) * logical_2q_state(hs, 0, 1).dag()
        model.add_observable(
            rho, outfile='10_01_coherence.dat', exp_unit='dimensionless',
            time_unit=time_unit, col_label='|10><01|', is_real=False)
    elif n_nodes == 1:
        for i in [0, 1]:
            ket = logical_1q_state(hs, i)
            rho = ket * ket.dag()
            model.add_observable(
                rho, outfile='qubit_pop.dat', exp_unit='dimensionless',
                time_unit=time_unit, col_label='P(%d)' % i,
                is_real=True)
    else:
        raise ValueError("Invalid number of nodes")
    for ls in hs.local_factors:
        n_op = convert_to_qutip(Create(hs=ls) * Destroy(hs=ls),
                                full_space=hs)
        model.add_observable(
            n_op, outfile='excitation.dat', exp_unit='dimensionless',
            time_unit=time_unit, col_label=ls.label, is_real=True)

    model.user_data['time_unit'] = time_unit
    model.user_data['write_jump_record'] = 'jump_record.dat'
    model.user_data['write_final_state'] = 'psi_final.dat'
    return model


def make_qdyn_oct_model(
        network_slh, num_vals, controls, *, oct_target, energy_unit='MHz',
        mcwf=False, non_herm=False, use_dissipation_superop=True,
        lambda_a=1e-5, seed=None, allow_negative=True, **kwargs):
    """Construct a QDYN level model for OCT, for two or more nodes"""
    import logging
    logger = logging.getLogger(__name__)
    hs = network_slh.H.space
    labels = [ls.label for ls in hs.local_factors]
    n_nodes = len([l for l in labels if l.startswith('q')])
    assert 1 < n_nodes < 10
    psi00 = logical_2q_state(hs, 0, 0)
    psi01 = logical_2q_state(hs, 0, 1)
    psi10 = logical_2q_state(hs, 1, 0)
    psi11 = logical_2q_state(hs, 1, 1)
    dicke_1 = dicke_state(hs, excitations=1)
    dicke_init_half = dicke_init_state(hs, excitations=(n_nodes//2))
    dicke_init_full = dicke_init_state(hs, excitations=(n_nodes))
    #dicke_half = dicke_state(hs, excitations=(n_nodes//2))  # TODO
    states = OrderedDict(
        [('00', psi00), ('01', psi01), ('10', psi10), ('11', psi11),
         ('dicke_init_half', dicke_init_half),
         ('dicke_init_full', dicke_init_full), ('dicke_1', dicke_1)])

    model = make_qdyn_model(
        network_slh, num_vals, controls, energy_unit=energy_unit, mcwf=mcwf,
        non_herm=non_herm, use_dissipation_superop=use_dissipation_superop,
        states=states)

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

    if oct_target == 'excitation_transfer_fw':
        model.user_data['initial_states'] = '10'
        model.user_data['target_states'] = '01'
    elif oct_target == 'dicke_1':
        model.user_data['initial_states'] = '10'
        model.user_data['target_states'] = 'dicke_1'
    elif oct_target == 'dicke_init_half':
        model.user_data['initial_states'] = '00'
        model.user_data['target_states'] = 'dicke_init_half'
    elif oct_target == 'dicke_init_full':
        model.user_data['initial_states'] = '00'
        model.user_data['target_states'] = 'dicke_init_half'
    elif oct_target == 'dicke_half':
        model.user_data['initial_states'] = 'dicke_init_half'
        model.user_data['target_states'] = 'dicke_half'
    elif oct_target == 'excite_first_qubit':
        model.user_data['initial_states'] = '00'
        model.user_data['target_states'] = '10'
    elif oct_target == 'excitation_transfer_bw':
        model.user_data['initial_states'] = '01'
        model.user_data['target_states'] = '10'
    elif oct_target == 'gate':
        model.user_data['basis'] = '00,01,10,11'
        model.user_data['gate'] = 'target_gate.dat'
    else:
        raise ValueError("Unknown oct_target %s" % oct_target)

    if seed is not None:
        model.user_data['seed'] = seed

    return model


def make_qdyn_oct_single_node_model(
        network_slh, num_vals, controls, *, oct_target, energy_unit='MHz',
        mcwf=False, non_herm=False, use_dissipation_superop=True,
        lambda_a=1e-5, seed=None, nodiss=False, **kwargs):
    """Construct a QDYN level model for OCT for control tasks on a single
    node"""
    hs = network_slh.H.space
    psi0 = logical_1q_state(hs, 0)
    psi1 = logical_1q_state(hs, 1)
    states = OrderedDict([('0', psi0), ('1', psi1)])

    model = make_qdyn_model(
        network_slh, num_vals, controls, energy_unit=energy_unit, mcwf=mcwf,
        non_herm=non_herm, use_dissipation_superop=use_dissipation_superop,
        states=states, nodiss=nodiss)

    for i, pulse in enumerate(model.pulses()):
        pulse.config_attribs.update(OrderedDict([
            ('oct_shape', 'flattop'),
            ('shape_t_start', pulse.t0), ('shape_t_stop', pulse.T),
            ('t_rise', 0.1*pulse.T), ('t_fall', 0.1*pulse.T),
            ('oct_lambda_a', lambda_a), ('oct_increase_factor', 5),
            ('oct_outfile', 'pulse%d.oct.dat' % (i+1)),
            ('oct_pulse_max', UnitFloat(500, 'MHz')),
            ('oct_pulse_min',  UnitFloat(-500, 'MHz')),
        ]))

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

    if oct_target == 'excite_qubit':
        model.user_data['initial_states'] = '0'
        model.user_data['target_states'] = '1'
    elif oct_target == 'gate':
        model.user_data['basis'] = '0,1'
        model.user_data['gate'] = 'target_gate.dat'
    else:
        raise ValueError("Unknown oct_target %s" % oct_target)

    if seed is not None:
        model.user_data['seed'] = seed

    return model


def prepare_rf_prop(model, rf_oct, *rf_prop, oct_pulses='pulse*.oct.dat'):
    """Generate propagation runfolders by writing model to every propagation
    runfolder and copying the optimized pulses from rf_oct"""
    for rf in rf_prop:
        model.write_to_runfolder(rf)
        for file in glob(join(rf_oct, oct_pulses)):
            shutil.copy(file, rf)
