"""Wrappers for clusterjob-enabled pydoit tasks"""
import os
from textwrap import dedent
from os.path import join

import clusterjob


def submit_optimization(rf, n_trajs, task):
    """Asynchronously run optimization"""
    body = dedent(r'''
    {module_load}

    cd {rf}
    OMP_NUM_THREADS=1 mpirun -n {n_trajs} qdyn_optimize --n-trajs={n_trajs} \
         --J_T=J_T_sm .
    ''')
    taskname = task.name.replace(":", '_').replace(r'/', '_')
    jobscript = clusterjob.JobScript(
        body=body, filename=join(rf, 'oct.slr'),
        jobname=taskname, nodes=1, ppn=int(n_trajs), threads=1,
        stdout=join(rf, 'oct.log'))
    jobscript.rf = rf
    jobscript.n_trajs = str(int(n_trajs))
    run = jobscript.submit(cache_id=taskname)
    run.dump(join(rf, 'oct.job.dump'))


def submit_crab(rf, n_crab, n_trajs, seed, complex, task):
    """Asynchronously run optimization"""
    if complex:
        body = dedent(r'''
        {module_load}

        python -u ./run_crab_simplex.py --n-crab={n_crab} --n-trajs={n_trajs} \
            --complex --seed={seed} {rf}
        ''')
    else:
        body = dedent(r'''
        {module_load}

        python -u ./run_crab_simplex.py --n-crab={n_crab} --n-trajs={n_trajs} \
            --seed={seed}} {rf}
        ''')
    taskname = task.name.replace(":", '_').replace(r'/', '_')
    jobscript = clusterjob.JobScript(
        body=body, filename=join(rf, 'crab.slr'),
        jobname=taskname, nodes=1, ppn=int(n_trajs), threads=1,
        stdout=join(rf, 'crab.log'))
    jobscript.rf = rf
    jobscript.n_trajs = str(int(n_trajs))
    jobscript.n_crab = str(int(n_crab))
    jobscript.seed = str(int(seed))
    run = jobscript.submit(cache_id=taskname)
    run.dump(join(rf, 'crab.job.dump'))


def submit_crab_powell(rf, n_crab, n_trajs, seed, complex, task):
    """Asynchronously run optimization, using Powell's method instead of
    Nelder-Mead"""
    if complex:
        body = dedent(r'''
        {module_load}

        python -u ./run_crab_simplex.py --n-crab={n_crab} --n-trajs={n_trajs} \
            --complex --seed={seed} --method=Powell {rf}
        ''')
    else:
        body = dedent(r'''
        {module_load}

        python -u ./run_crab_simplex.py --n-crab={n_crab} --n-trajs={n_trajs} \
            --seed={seed}} --method=Powell {rf}
        ''')
    taskname = task.name.replace(":", '_').replace(r'/', '_')
    jobscript = clusterjob.JobScript(
        body=body, filename=join(rf, 'crab.slr'),
        jobname=taskname, nodes=1, ppn=int(n_trajs), threads=1,
        stdout=join(rf, 'crab.log'))
    jobscript.rf = rf
    jobscript.n_trajs = str(int(n_trajs))
    jobscript.n_crab = str(int(n_crab))
    jobscript.seed = str(int(seed))
    run = jobscript.submit(cache_id=taskname)
    run.dump(join(rf, 'crab.job.dump'))


def submit_propagation(rf, n_trajs):
    """Run propagation"""
    body = dedent(r'''
    {module_load}

    cd {rf}
    OMP_NUM_THREADS=1 mpirun -n {n_trajs} qdyn_prop_traj --n-trajs={n_trajs} \
        --use-oct-pulses --write-final-state=state_final.dat .
    ''')
    taskname = task.name.replace(":", '_').replace(r'/', '_')
    jobscript = clusterjob.JobScript(
        body=body, filename=join(rf, 'prop.slr'),
        jobname=taskname, nodes=1, ppn=int(n_trajs), threads=1,
        stdout=join(rf, 'prop.log'))
    jobscript.rf = rf
    jobscript.n_trajs = str(int(n_trajs))
    run = jobscript.submit(cache_id=taskname, force=True)
    run.dump(join(rf, 'prop.job.dump'))


def wait_for_clusterjob(dumpfile):
    """Wait until the clusterjob.AsyncResult cached in the given dumpfile
    ends"""
    try:
        run = clusterjob.AsyncResult.load(dumpfile)
        run.wait()
        os.unlink(dumpfile)
        return run.successful()
    except OSError:
        pass
