"""Wrappers for actions in pydoit tasks"""
import os
from glob import glob
from os.path import join
import subprocess

import numpy as np
import psutil
import QDYN
import clusterjob


def run_traj_oct(rf, n_trajs, wait=False):
    """Asynchronously run optimization, storing the pid of the oct process"""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    cmd = ['mpirun', '-n', str(int(n_trajs)), 'qdyn_optimize',
           '--n-trajs=%s' % int(n_trajs), '--J_T=J_T_sm', '.']
    with open(join(rf, 'oct.log'), 'wb') as log_fh:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, cwd=rf)
        if wait:
            proc.wait()
        else:
            with open(join(rf, 'oct.pid'), 'w') as pid_fh:
                pid_fh.write(str(proc.pid))


def run_traj_prop(
        rf, n_trajs, n_procs=None, use_oct_pulses=True, config='config'):
    """Run propagation"""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    if n_procs is None:
        n_procs = int(n_trajs)
    if use_oct_pulses:
        use_oct_pulses_arg = ['--use-oct-pulses']
    else:
        use_oct_pulses_arg = []
    if n_procs > 1:
        cmd = (['mpirun', '-n', str(n_procs), 'qdyn_prop_traj',
                '--n-trajs=%s' % int(n_trajs)] + use_oct_pulses_arg +
               ['--config=%s' % config, '.'])
    else:
        cmd = (['qdyn_prop_traj', ] + use_oct_pulses_arg +
               ['--config=%s' % config, '.'])
    files_to_delete = glob(join(rf, 'state_final.dat.*'))
    files_to_delete.extend(glob(join(rf, 'jump_record.dat.*')))
    for file in files_to_delete:
        try:
            os.unlink(file)
        except FileNotFoundError:
            pass
    with open(join(rf, 'prop.log'), 'wb') as log_fh:
        subprocess.call(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, cwd=rf)


def wait_for_oct(rf):
    """Wait until the OCT process with the given pid ends"""
    try:
        with open(join(rf, 'oct.pid')) as pid_fh:
            pid = int(pid_fh.read())
            proc = psutil.Process(pid)
            proc.wait()
    except (psutil.NoSuchProcess, OSError):
        pass


def update_config(rf, lambda_a=None, iter_stop=None, J_T_conv=None):
    """Update the config file in the given runfolder with the given new OCT
    parameters. For any parameter passed as None, the original value in the
    config is kept unchanged"""
    config = QDYN.config.read_config_file(join(rf, 'config'))
    if iter_stop is None:
        config['oct']['iter_stop'] = iter_stop
    if J_T_conv is not None:
        config['oct']['J_T_conv'] = J_T_conv
    if lambda_a is not None:
        for pulse_config in config['pulse']:
            pulse_config['oct_lambda_a'] = lambda_a
    QDYN.config.write_config(config, join(rf, 'config'))


def wait_for_clusterjob(dumpfile):
    """Wait until the clusterjob.AsyncResult cached in the given dumpfile
    ends
    """
    try:
        run = clusterjob.AsyncResult.load(dumpfile)
        run.wait()
        os.unlink(dumpfile)
        return run.successful()
    except OSError:
        pass


def write_rho_prop_custom_config(rf, oct_iter, config_out):
    """Write `config_out` in `rf` for propagation of the optimized pulses
    ``<oct_outfile>.<oct_iter>``, based on the template in ``rf/config_rho``
    """
    config = QDYN.config.read_config_file(join(rf, 'config_rho'))
    if 'oct' in config:
        # we wouldn't want to accidentally use this config file for OCT
        del config['oct']
    for pulse_config in config['pulse']:
        pulse_file = "%s.%08d" % (pulse_config['oct_outfile'], int(oct_iter))
        # we wouldn't want to accidentally propagate the final optimized pulse
        if 'oct_outfile' in pulse_config:
            del pulse_config['oct_outfile']
        assert os.path.isfile(join(rf, pulse_file))
        pulse_config['filename'] = pulse_file
    for obs_config in config['observables']:
        outfile = obs_config['outfile']
        if outfile == 'P_target.dat':
            obs_config['outfile'] = "%s.%08d" % (outfile, int(oct_iter))
            obs_config['from_time_index'] = -1
            obs_config['to_time_index'] = -1
            config['observables'] = [obs_config, ]
            break
    for key in ['write_jump_record', 'write_final_state']:
        if key in config['user_strings']:
            del config['user_strings'][key]
    assert len(config['observables']) == 1
    assert config['user_logicals']['rho']
    QDYN.config.write_config(config, join(rf, config_out))


def _sort_key_oct_iters(f):
    """Key for sorting files according to iteration number (file extension)"""
    try:
        a, b = f.rsplit('.', maxsplit=1)
        return (a, int(b))
    except ValueError:
        return (f, 0)


def collect_rho_prop_errors(*P_target_obs_files, outfile):
    """Combine the expectation values for the target projector for different
    OCT iterations into a single data file (`outfile`)"""
    T = None
    with open(outfile, 'w') as out_fh:
        out_fh.write("#%9s%25s\n" % ('iter', 'Error 1-<P>_rho'))
        for obs_file in sorted(P_target_obs_files, key=_sort_key_oct_iters):
            n_iter = int(obs_file.rsplit('.', maxsplit=1)[-1])
            data = np.genfromtxt(obs_file)
            assert data.shape == (3, )
            if T is None:
                T = data[0]
                # we want to make sure all files have the same final time
            assert abs(data[0] - T) < 1e-14
            out_fh.write("%10d%25.16E\n" % (n_iter, 1.0 - data[1]))
