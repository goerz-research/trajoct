"""Wrappers for command lines tools"""
import os
import subprocess

def env(**kwargs):
    """Return a modified shell environement"""
    _env = os.environ.copy()
    for key, val in kwargs.items():
        _env[key] = str(val)
    return _env


def make_sh_cmd(cmd):
    """Create a subprocess wrapper around cmd"""

    def sh_cmd(args=None, _env=None, _out=None):
        """Run cmd with args"""
        if args is None:
            args = []
        if not isinstance (args, (list, tuple)):
            args = [args, ]
        if _env is None:
            _env = env(OMP_NUM_THREADS=1)
        if _out is None:
            _out = subprocess.PIPE
        full_cmd = cmd + args
        if isinstance(_out, str):
            with open(_out, 'w') as out_fh:
                return subprocess.Popen(full_cmd, env=_env, stdout=out_fh,
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
        else:
            return subprocess.Popen(full_cmd, env=_env, stdout=_out,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
    return sh_cmd


_qdyn_optimize = make_sh_cmd(
        ['qdyn_optimize', '--internal-units=GHz_units.txt'])


def qdyn_optimize(args=None, _env=None, _out=None):
    assert '--J_T=' in " ".join(args)
    return _qdyn_optimize(args=args, _env=_env, _out=_out)


qdyn_prop_traj = make_sh_cmd(
        ['qdyn_prop_traj', '--internal-units=GHz_units.txt'])

qdyn_check_config = make_sh_cmd(
        ['qdyn_check_config',])
