"""Tests of basic functionality"""
from textwrap import dedent
from shwrapper import qdyn_check_config
import pytest
from copy import copy
from collections import OrderedDict
import single_sided_network
import qdyn_model
import numpy as np
from sympy import Symbol
import QDYN
from QDYN.pulse import blackman



@pytest.fixture
def num_vals_no_fb():
    Delta =  5000.0  # MHz
    g     =    50.0  # MHz
    kappa =     0.5  # MHz
    num_vals = {
        Symbol('Delta_1', real=True):   Delta,
        Symbol('Delta_2', real=True):   Delta,
        Symbol('g_2', positive=True):   g,
        Symbol('g_1', positive=True):   g,
        Symbol('kappa', positive=True): kappa,
    }
    return num_vals


@pytest.fixture
def controls_no_fb():
    t0    = -4    # microsec
    T     =  4    # microsec
    nt    = 2001
    E0    =    70.0  # MHz
    tgrid = QDYN.pulse.pulse_tgrid(t0=t0, T=T, nt=nt) # microsec
    p = QDYN.pulse.Pulse(
        tgrid, amplitude=(E0 * blackman(tgrid, t0, T)),
        time_unit='microsec', ampl_unit='MHz')
    controls = OrderedDict([
        (Symbol('Omega_1'), p.copy()),
        (Symbol('Omega_2'), p.copy()),
    ])
    return controls


@pytest.fixture
def model_no_fb(num_vals_no_fb, controls_no_fb):
    num_vals = num_vals_no_fb
    controls = controls_no_fb
    slh = single_sided_network.network_slh(
        n_cavity=5, n_nodes=2, topology='open')
    model = qdyn_model.make_qdyn_model(
        slh, num_vals, controls, energy_unit='MHz',
        mcwf=False, non_herm=False, states=None)
    return model


@pytest.fixture
def model_no_fb_oct_fw(num_vals_no_fb, controls_no_fb):
    num_vals = num_vals_no_fb
    controls = controls_no_fb
    slh = single_sided_network.network_slh(
        n_cavity=5, n_nodes=2, topology='open')
    model = qdyn_model.make_qdyn_oct_model(
        slh, num_vals, controls, energy_unit='MHz',
        mcwf=False, non_herm=False, oct_target='excitation_transfer_fw')
    return model


@pytest.fixture
def model_no_fb_oct_gate(num_vals_no_fb, controls_no_fb):
    num_vals = num_vals_no_fb
    controls = controls_no_fb
    slh = single_sided_network.network_slh(
        n_cavity=5, n_nodes=2, topology='open')
    model = qdyn_model.make_qdyn_oct_model(
        slh, num_vals, controls, energy_unit='MHz',
        mcwf=False, non_herm=False, oct_target='sqrt_SWAP')
    return model


def test_no_feedback_system(model_no_fb, tmpdir):
    model = copy(model_no_fb)
    model.write_to_runfolder(str(tmpdir))
    config = tmpdir.join('config').read().strip()
    expected = dedent(r'''
    tgrid: t_start = -4_microsec, t_stop = 4_microsec, nt = 2001

    prop: method = newton, use_mcwf = F

    pulse: type = file, time_unit = microsec, ampl_unit = MHz
    * id = 1, filename = pulse1.dat
    * id = 2, filename = pulse2.dat

    ham: type = matrix, real_op = F, n_surf = 100, sparsity_model = indexed
    * filename = H0.dat, op_unit = energy_unit, op_type = potential
    * filename = H1.dat, op_unit = dimensionless, op_type = dipole, pulse_id = 1
    * filename = H2.dat, op_unit = dimensionless, op_type = dipole, pulse_id = 2

    dissipator:
    * type = lindblad_ops, filename = L1.dat, real_op = F, op_unit = sqrt_MHz, &
      sparsity_model = indexed

    user_strings: time_unit = microsec, write_jump_record = jump_record.dat, &
      write_final_state = psi_final.dat
    ''').strip()
    assert config == expected
    p = qdyn_check_config(str(tmpdir.join('config')))
    p.wait()
    out = p.stdout.read()
    assert 'Successfully read config file' in out


def test_no_feedback_oct_fw_system(model_no_fb_oct_fw, tmpdir):
    model = copy(model_no_fb_oct_fw)
    model.write_to_runfolder(str(tmpdir))
    config = tmpdir.join('config').read().strip()
    expected = dedent(r'''
    tgrid: t_start = -4_microsec, t_stop = 4_microsec, nt = 2001

    prop: method = newton, use_mcwf = F

    pulse: oct_shape = flattop, shape_t_start = -4_microsec, &
      shape_t_stop = 4_microsec, t_rise = 0.4_microsec, t_fall = 0.4_microsec, &
      oct_lambda_a = 1e-05, oct_increase_factor = 5, oct_pulse_max = 500_MHz, &
      oct_pulse_min = -500_MHz, type = file, time_unit = microsec, ampl_unit = MHz
    * id = 1, filename = pulse1.dat, oct_outfile = pulse1.oct.dat
    * id = 2, filename = pulse2.dat, oct_outfile = pulse2.oct.dat

    ham: type = matrix, real_op = F, n_surf = 100, sparsity_model = indexed
    * filename = H0.dat, op_unit = energy_unit, op_type = potential
    * filename = H1.dat, op_unit = dimensionless, op_type = dipole, pulse_id = 1
    * filename = H2.dat, op_unit = dimensionless, op_type = dipole, pulse_id = 2

    dissipator:
    * type = lindblad_ops, filename = L1.dat, real_op = F, op_unit = sqrt_MHz, &
      sparsity_model = indexed

    psi:
    * type = file, filename = psi_00.dat, label = 00

    psi:
    * type = file, filename = psi_01.dat, label = 01

    psi:
    * type = file, filename = psi_10.dat, label = 10

    psi:
    * type = file, filename = psi_11.dat, label = 11

    oct: method = krotovpk, J_T_conv = 0.0001, max_ram_mb = 100000, &
      iter_dat = oct_iters.dat, iter_stop = 100, keep_pulses = all, &
      limit_pulses = T, params_file = oct_params.dat, tau_dat = oct_tau.dat

    user_strings: time_unit = microsec, write_jump_record = jump_record.dat, &
      write_final_state = psi_final.dat, initial_states = 10, target_states = 01
    ''').strip()
    assert config == expected
    p = qdyn_check_config(str(tmpdir.join('config')))
    p.wait()
    out = p.stdout.read()
    assert 'Successfully read config file' in out


def test_no_feedback_oct_gate_system(model_no_fb_oct_gate, tmpdir):
    model = copy(model_no_fb_oct_gate)
    model.write_to_runfolder(str(tmpdir))
    QDYN.gate2q.sqrt_SWAP.write(str(tmpdir.join('target_gate.dat')))
    config = tmpdir.join('config').read().strip()
    expected = dedent(r'''
    tgrid: t_start = -4_microsec, t_stop = 4_microsec, nt = 2001

    prop: method = newton, use_mcwf = F

    pulse: oct_shape = flattop, shape_t_start = -4_microsec, &
      shape_t_stop = 4_microsec, t_rise = 0.4_microsec, t_fall = 0.4_microsec, &
      oct_lambda_a = 1e-05, oct_increase_factor = 5, oct_pulse_max = 500_MHz, &
      oct_pulse_min = -500_MHz, type = file, time_unit = microsec, ampl_unit = MHz
    * id = 1, filename = pulse1.dat, oct_outfile = pulse1.oct.dat
    * id = 2, filename = pulse2.dat, oct_outfile = pulse2.oct.dat

    ham: type = matrix, real_op = F, n_surf = 100, sparsity_model = indexed
    * filename = H0.dat, op_unit = energy_unit, op_type = potential
    * filename = H1.dat, op_unit = dimensionless, op_type = dipole, pulse_id = 1
    * filename = H2.dat, op_unit = dimensionless, op_type = dipole, pulse_id = 2

    dissipator:
    * type = lindblad_ops, filename = L1.dat, real_op = F, op_unit = sqrt_MHz, &
      sparsity_model = indexed

    psi:
    * type = file, filename = psi_00.dat, label = 00

    psi:
    * type = file, filename = psi_01.dat, label = 01

    psi:
    * type = file, filename = psi_10.dat, label = 10

    psi:
    * type = file, filename = psi_11.dat, label = 11

    oct: method = krotovpk, J_T_conv = 0.0001, max_ram_mb = 100000, &
      iter_dat = oct_iters.dat, iter_stop = 100, keep_pulses = all, &
      limit_pulses = T, params_file = oct_params.dat, tau_dat = oct_tau.dat

    user_strings: time_unit = microsec, write_jump_record = jump_record.dat, &
      write_final_state = psi_final.dat, basis = 00\,01\,10\,11, &
      gate = target_gate.dat
    ''').strip()
    assert config == expected
    p = qdyn_check_config(str(tmpdir.join('config')))
    p.wait()
    out = p.stdout.read()
    assert 'Successfully read config file' in out
    gate = QDYN.gate2q.Gate2Q.read(str(tmpdir.join('target_gate.dat')))
    diff = np.abs(QDYN.linalg.vectorize(gate - QDYN.gate2q.sqrt_SWAP))
    assert QDYN.linalg.norm(diff) < 1e-14


def test_logical_2q_state():
    """Test construction of logical states"""
    from qdyn_model import logical_2q_state, state
    import single_sided_network

    slh = single_sided_network.network_slh(
        n_cavity=5, n_nodes=2, topology='open')
    hs = slh.H.space

    assert logical_2q_state(hs, 0, 0) == state(hs, 0, 0, 0, 0)
    assert logical_2q_state(hs, 0, 1) == state(hs, 0, 0, 1, 0)
    assert logical_2q_state(hs, 1, 0) == state(hs, 1, 0, 0, 0)
    assert logical_2q_state(hs, 1, 1) == state(hs, 1, 0, 1, 0)