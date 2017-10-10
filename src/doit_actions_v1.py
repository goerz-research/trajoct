"""Wrappers for actions in pydoit tasks"""
import os
from glob import glob
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


def run_traj_prop(rf, n_trajs, n_procs=None, use_oct_pulses=True):
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
               ['--write-final-state=state_final.dat', '.'])
    else:
        cmd = (['qdyn_prop_traj', ] + use_oct_pulses_arg +
               ['--write-final-state=state_final.dat', '.'])
    files_to_delete = glob(join(rf, 'state_final.dat.*'))
    files_to_delete.extend(glob(join(rf, 'jump_record.dat.*')))
    for file in files_to_delete:
        os.unlink(file)
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
