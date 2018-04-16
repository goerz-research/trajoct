import QDYN
import numpy as np
from scipy.signal import savgol_filter


def smooth_pulse_savgol(pulse, **kwargs):
    filtered = pulse.copy()
    ampl_filtered = savgol_filter(pulse.amplitude.real, **kwargs)
    filtered.amplitude = ampl_filtered
    return filtered


def pulse_delta_smoothing(pulses, smooth_pulse, **kwargs):
    res = []
    for pulse in pulses:
        delta = pulse.copy()
        edge = 0.05 * float(pulse.T)
        window = QDYN.pulse.flattop(
            pulse.tgrid, float(pulse.t0) + edge,
            float(pulse.T) - edge, 0)
        delta.amplitude -= smooth_pulse(pulse, **kwargs).amplitude
        delta.amplitude = window * np.abs(delta.amplitude)
        res.append(delta)
    return res
