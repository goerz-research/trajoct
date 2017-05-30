#!/usr/bin/env python
"""Script for simplex optimization"""
import sys
from os.path import join
from functools import partial

import QDYN
from QDYN.pulse import flattop, blackman
from QDYN.pulse import CRAB_carrier
from QDYN.analytical_pulse import AnalyticalPulse
from QDYN.units import UnitFloat, UnitConvert

from scipy.optimize import minimize
import numpy as np
import click

from src.doit_actions_v1 import run_traj_prop
from src.dicke_half_model_v2 import err_dicke_half


def update_pulses(analytical_pulses, x):
    """Distribute the parameter array `x` to the list of `analytical_pulses`"""
    offset = 0
    for pulse in analytical_pulses:
        n = len(pulse.parameters_to_array())
        pulse.array_to_parameters(x[offset:offset+n])
        offset += n


def generate_fun(rf, analytical_pulses, n_trajs, pulse_tgrid):
    """Return the function `fun` to be minimized

    The function has two side-effects:
    * Modify data of `analytical_pulses`
    * Write data to runfolder `rf`
    """

    def fun(x):
        """Function to be minized, return the error with with a N/2 Dicke state
        is realized"""
        update_pulses(analytical_pulses, x)
        for pulse in analytical_pulses:
            assert 'filename' in pulse.config_attribs
            filename = pulse.config_attribs['filename']
            pulse.to_num_pulse(pulse_tgrid).write(join(rf, filename))
        run_traj_prop(rf, n_trajs)
        res = err_dicke_half(
            rf, final_states_glob='state_final.dat*', rho=False)
        print("%s -> %s" % (x, res))
        return res

    return fun


def generate_crab_amplitude(
        A, w_n, time_unit, t0, T, oct_pulse_min, oct_pulse_max, complex=True):
    """Generate a an amplitude formula `crab_amplitude` suitable as a formula
    for AnalyticalPulse

    Args:
        A (float): Amplitude renormalization factor
        w_n (ndarray): array of basis frequencies, in "internal units"
        time_unit (str): The unit of the time grid values that will be passed
            to `crab_amplitude`
        t0 (float): The start of the time evolution
        T (float): The end of the time evolution
        oct_pulse_min (float): The maximum allowed pulse amplitude
        oct_pulse_max (float): The minimum allowed pulse amplitude. Pulse
            values will be clipped between `oct_pulse_min` and `oct_pulse_max`.
            This has no effect for complex pulses.
    """

    def crab_amplitude(tgrid, a_n, b_n, d):
        """Amplitude formula for AnalyticalPulse"""
        shape_t_start = max(t0, t0 + 0.9 * np.tanh(0.1*d) * (T - t0))
        shape_t_stop = min(T, T + 0.9 * np.tanh(0.1*d) * (T - t0))
        Delta_T = shape_t_stop - shape_t_start
        compress = (T - t0) / Delta_T
        return np.clip(
            a=(A *
               CRAB_carrier(
                   (t0 + compress * (tgrid - shape_t_start)),
                   time_unit=time_unit, freq=w_n, freq_unit='iu', a=a_n,
                   b=b_n, complex=complex) *
               blackman(
                   tgrid, t_start=shape_t_start, t_stop=shape_t_stop)),
            a_min=oct_pulse_min, a_max=oct_pulse_max)

    return crab_amplitude


def generate_CRAB_guess(rf, n_crab, complex=True):
    """Generate guess pulses

    Returns:
        tuple: (analytical_pulses, pulse_tgrid, x0)
            where `analytical_pulses` is a list of AnalyticalPulse instances,
            `pulse_tgrid` is the time grid for converting analytical pulses to
            numerical pulses, and `x0` is an array of the concatenated control
            parameters for all the `analytical_pulses`
    """
    unit_convert = UnitConvert()
    config_data = QDYN.config.read_config_file(join(rf, 'config'))
    pulse_tgrid = QDYN.pulse.pulse_tgrid(
            T=config_data['tgrid']['t_stop'],
            nt=config_data['tgrid']['nt'],
            t0=config_data['tgrid']['t_start'])
    time_unit = config_data['pulse'][0]['time_unit']
    ampl_unit = config_data['pulse'][0]['ampl_unit']
    t0 = config_data['tgrid']['t_start']
    T = config_data['tgrid']['t_stop']
    Delta_T = float(UnitFloat(T - t0, time_unit))
    c = unit_convert.convert(1, time_unit, 'iu')
    w0 = np.pi * c / Delta_T
    harmonics = w0 * np.arange(1, n_crab+1)
    w_n = harmonics + w0 * (np.random.random(n_crab) - 0.5)
    analytical_pulses = []
    x0 = []
    for pulse_data in config_data['pulse']:
        assert pulse_data['type'] == 'file'
        assert pulse_data['time_unit'] == time_unit
        assert pulse_data['ampl_unit'] == ampl_unit
        E_0 = UnitFloat(pulse_data['E_0'], ampl_unit)
        oct_pulse_min = UnitFloat(pulse_data['oct_pulse_min'], ampl_unit)
        oct_pulse_max = UnitFloat(pulse_data['oct_pulse_max'], ampl_unit)
        parameters = {
            'a_n': 2*np.random.random(n_crab) - 1.0,
            'b_n': 2*np.random.random(n_crab) - 1.0,
            'd': 0.0,
        }
        x0.extend(parameters['a_n'])
        x0.extend(parameters['b_n'])
        x0.append(parameters['d'])
        generate_ampl = partial(
            generate_crab_amplitude, w_n=w_n, time_unit=time_unit,
            t0=float(t0), T=float(T),
            oct_pulse_min=float(oct_pulse_min),
            oct_pulse_max=float(oct_pulse_max),
            complex=complex)
        crab_amplitude = generate_ampl(A=1.0)
        E_max = np.max(np.abs(crab_amplitude(
            pulse_tgrid, parameters['a_n'], parameters['b_n'],
            parameters['d'])))
        crab_pulse = AnalyticalPulse.from_func(
            func=generate_ampl(A=(float(E_0)/E_max)),
            parameters=parameters, time_unit=time_unit, ampl_unit=ampl_unit,
            config_attribs=pulse_data.copy())
        analytical_pulses.append(crab_pulse)
    return analytical_pulses, pulse_tgrid, np.array(x0)


def run_crab_simplex(rf, n_crab, n_trajs, complex=True, method='Nelder-Mead'):
    """Run a simplex optimization"""
    # set up guess
    analytical_pulses, pulse_tgrid, x0 = generate_CRAB_guess(
        rf=rf, n_crab=n_crab, complex=True)
    fun = generate_fun(
        rf=rf, analytical_pulses=analytical_pulses, n_trajs=n_trajs,
        pulse_tgrid=pulse_tgrid)
    for pulse in analytical_pulses:
        filename = pulse.config_attribs['filename'] + ".crab_guess"
        pulse.to_num_pulse(pulse_tgrid).write(join(rf, filename))
    # optimize
    res = minimize(fun, x0, method='Nelder-Mead')
    # write out optimized pulses
    update_pulses(analytical_pulses, res.x)
    for pulse in analytical_pulses:
        filename = pulse.config_attribs['oct_outfile']
        pulse.to_num_pulse(pulse_tgrid).write(join(rf, filename))


@click.command()
@click.option('--n-crab', type=int, default=10,
              help="number of CRAB components")
@click.option('--n-trajs', type=int, default=1,
              help="number of MPI trajectories")
@click.option('--seed', type=int, help="seed for random number generator used "
              "for randomized pulse frequencies")
@click.option('--complex', is_flag=True, help="Use complex pulses")
@click.option('--method', type=click.Choice(['Nelder-Mead', 'Powell']),
              default='Nelder-Mead', help="Optimization method to use")
@click.argument('rf', type=click.Path(exists=True))
def main(n_crab, n_trajs, seed, complex, method, rf):
    """Run a simplex optimization on the given runfolder RF"""
    if seed is not None:
        np.random.seed(seed)
    run_crab_simplex(
        rf=rf, n_crab=n_crab, n_trajs=n_trajs, complex=complex, method=method)


if __name__ == "__main__":
    sys.exit(main())
