"""Wrappers for actions in pydoit tasks"""
import os
from os.path import join
import subprocess

import psutil


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


def run_traj_prop(rf, n_trajs):
    """Run propagation"""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    cmd = ['mpirun', '-n', str(int(n_trajs)), 'qdyn_prop_traj',
           '--n-trajs=%s' % int(n_trajs), '--use-oct-pulses',
           '--write-final-state=state_final.dat', '.']
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
