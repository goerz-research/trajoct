"""Routines for constructing guess pulses"""
import QDYN
from QDYN.pulse import blackman


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
    return pulse


def zero_pulse(tgrid):
    """Return a Pulse with zero amplitude, with dimensionless units"""
    pulse = QDYN.pulse.Pulse(
        tgrid, amplitude=(0.0 * tgrid),
        time_unit='dimensionless', ampl_unit='dimensionless',
        freq_unit='dimensionless')
    return pulse
