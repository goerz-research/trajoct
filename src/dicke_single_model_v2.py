"""Dicke optimization for a single excitation"""
from os.path import join
from glob import glob

import qutip

import QDYN
from .qdyn_model_v2 import make_qdyn_oct_model, err_state_to_state

from .dicke_half_model_v2 import (
    pi_pulse, zero_pulse, num_vals, get_mu_1, dicke_guess_controls)

__all__ = [
    'pi_pulse', 'zero_pulse', 'num_vals', 'get_mu_1',
    'dicke_guess_controls', 'write_dicke_single_model']


def write_dicke_single_model(
        slh, rf, T, theta=0, E0_cycles=2, mcwf=False, non_herm=False,
        lambda_a=1.0, J_T_conv=1e-4, iter_stop=5000, nt=None,
        max_ram_mb=100000, kappa=0.01, seed=None, observables='all'):
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
        seed (int or None): seed for MCWF
        observables (str): One of 'all', 'last'

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
    qdyn_model = make_qdyn_oct_model(
        slh, num_vals=num_vals(theta=theta, n_nodes=n_nodes, kappa=kappa),
        controls=controls, energy_unit='dimensionless',
        mcwf=mcwf, non_herm=non_herm, oct_target='dicke_1',
        lambda_a=lambda_a, iter_stop=iter_stop, keep_pulses='prev',
        J_T_conv=J_T_conv, max_ram_mb=max_ram_mb, seed=seed,
        observables=observables)
    qdyn_model.user_data['state_label'] = '10'  # for prop
    if (mcwf, non_herm) == (False, False):
        qdyn_model.user_data['rho'] = True
    qdyn_model.write_to_runfolder(rf)


def err_dicke_single(rf, final_states_glob, rho=False):
    config = QDYN.config.read_config_file(join(rf, 'config'))
    n = int(config['ham'][0]['n_surf'])
    psi_dicke_single = QDYN.state.read_psi_amplitudes(
            join(rf, "psi_dicke_1.dat"), n)
    if rho:
        psi_dicke_single = qutip.Qobj(psi_dicke_single)
        rho_dicke_single = psi_dicke_single * psi_dicke_single.dag()
        final_state_files = glob(join(rf, final_states_glob))
        assert len(final_state_files) == 1
        final_rho_fname = final_state_files[0]
        final_rho = qutip.Qobj(QDYN.io.read_indexed_matrix(
            final_rho_fname, shape=(n, n)))
        err = 1.0 - (rho_dicke_single * final_rho).tr()
        assert err.imag < 1e-14
        return err.real
    else:
        return err_state_to_state(
            psi_dicke_single, join(rf, final_states_glob))
