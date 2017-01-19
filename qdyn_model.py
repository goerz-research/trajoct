"""Routines to construct the QDYN model for a multi-node network with linear
controls"""

from os.path import join
from glob import glob
from collections import OrderedDict
import shutil

import qutip
import QDYN
from QDYN.units import UnitFloat

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
    """Return logical two-qubit state |ij>"""
    assert i in [0, 1] and j in [0, 1]
    qnums = []
    found_q1 = False
    found_q1 = False
    for ls in space.local_factors:
        if ls.label == 'q1':
            qnums.append(i)
            found_q1 = True
        elif ls.label == 'q2':
            qnums.append(j)
            found_q2 = True
        else:
            qnums.append(0)
    assert found_q1 and found_q2
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


def make_qdyn_model(
        network_slh, num_vals, controls, energy_unit='MHz',
        mcwf=False, non_herm=False, states=None, add_observables=False):
    """Construct a QDYN LevelModel for a two-node network, configured for
    propagation

    Args:
        network_slh (SLH): SLH model of the system
        num_vals (dict): mapping of symbols to numerical values
        controls (OrderedDict): mapping of control symbols to Pulse instances.
        energy_unit (str): The physical unit of the drift Hamiltonian. The
            Lindbladians must be in units of sqrt-`energy_unit`.
        mcwf (bool): If True, initialize for MCWF propagation. Otherwise,
            non-dissipative Hilbert space or Liouville space propagation
        non_herm (bool): If True, add the decay term to the Hamiltonian. If
            used together with `mcwf=True`, this improves efficiency slightly.
            If used with `mwwf=False`, a non-Hermitian Hamiltonian would be
            propagated
        states (dict or None): dict label => state (numpy array of amplitudes).
            If None, no state will be added to the model.
        add_observables (bool): flag whether to add any observables to the
            config file. Only do this for propagation, not for OCT

    Notes:

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

    construct_mcwf_ham = False
    if mcwf:
        construct_mcwf_ham = True
    if non_herm:
        construct_mcwf_ham = False

    H_num = network_slh.H.substitute(num_vals)
    ham_parts = split_hamiltonian(H_num, use_cc=False, controls=control_syms)
    hs = H_num.space

    # drift Hamiltonian and dissipator
    H0 = (convert_to_qutip(ham_parts['H0'], full_space=hs) +
          convert_to_qutip(ham_parts['Hint'], full_space=hs))

    L = convert_to_qutip(network_slh.L[0, 0].substitute(num_vals),
                         full_space=hs)

    model = QDYN.model.LevelModel()

    if non_herm:
        H0 = H0 - 0.5j*(L.dag() * L)
    else:
        model.add_lindblad_op(L, op_unit='sqrt_%s' % energy_unit,
                              add_to_H_jump='indexed')
    model.add_ham(H0, op_unit=energy_unit, op_type='potential')

    for i, control_sym in enumerate(control_syms):
        H_d = convert_to_qutip(
            ham_parts['Hd_%d' % (i+1)].substitute({control_sym: 1}),
            full_space=hs)
        pulse = controls[control_sym]
        model.add_ham(H_d, pulse=pulse, op_unit='dimensionless',
                      op_type='dipole')

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
    model.set_propagation(T=T, nt=nt, t0=t0, time_unit=time_unit,
                          prop_method='newton', use_mcwf=mcwf, mcwf_order=2,
                          construct_mcwf_ham=construct_mcwf_ham)
    if states is not None:
        for label, psi in states.items():
            assert psi.shape[0] == H0.shape[0]
            model.add_state(psi, label)

    if add_observables:
        model.add_observable(
            L.dag()*L, outfile='darkstate_cond.dat', exp_unit='dimensionless',
            time_unit='microsec', col_label='<L^+L>', is_real=True)
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            ket = logical_2q_state(hs, i, j)
            rho = ket * ket.dag()
            model.add_observable(
                rho, outfile='qubit_pop.dat', exp_unit='dimensionless',
                time_unit='microsec', col_label='P(%d%d)' % (i, j),
                is_real=True)

    model.user_data['time_unit'] = time_unit
    model.user_data['write_jump_record'] = 'jump_record.dat'
    model.user_data['write_final_state'] = 'psi_final.dat'
    return model


def make_qdyn_oct_model(
        network_slh, num_vals, controls, *, oct_target, energy_unit='MHz',
        mcwf=False, non_herm=False, lambda_a=1e-5, seed=None, **kwargs):
    """Construct a QDYN level model for OCT"""
    hs = network_slh.H.space
    psi00 = logical_2q_state(hs, 0, 0)
    psi01 = logical_2q_state(hs, 0, 1)
    psi10 = logical_2q_state(hs, 1, 0)
    psi11 = logical_2q_state(hs, 1, 1)
    states = OrderedDict([('00', psi00), ('01', psi01), ('10', psi10),
                          ('11', psi11)])

    model = make_qdyn_model(
        network_slh, num_vals, controls, energy_unit=energy_unit, mcwf=mcwf,
        non_herm=non_herm, states=states)

    pulse_settings = {}
    for i, pulse in enumerate(model.pulses()):
        pulse_setting = OrderedDict([
            ('oct_shape', 'flattop'),
            ('shape_t_start', pulse.t0), ('shape_t_stop', pulse.T),
            ('t_rise', 0.1*pulse.T), ('t_fall', 0.1*pulse.T),
            ('oct_lambda_a', lambda_a), ('oct_increase_factor', 5),
            ('oct_outfile', 'pulse%d.oct.dat' % (i+1)),
            ('oct_pulse_max', UnitFloat(500, 'MHz')),
            ('oct_pulse_min',  UnitFloat(-500, 'MHz')),
        ])
        pulse_settings[pulse] = pulse_setting

    oct_settings = OrderedDict([
        ('pulse_settings', pulse_settings),
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
    elif oct_target == 'excitation_transfer_bw':
        model.user_data['initial_states'] = '01'
        model.user_data['target_states'] = '10'
    elif oct_target == 'sqrt_SWAP':
        model.user_data['basis'] = '00,01,10,11'
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