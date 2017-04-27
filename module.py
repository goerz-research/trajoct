"""This file is supposed to be source (`%run -i module.py` in IPython) in order
to make modules (http://modules.sourceforge.net) available viad the `module`
function.

This may then be used as e.g.

    >>> module('load', 'intel/xe2011')
    >>> module('list')
"""
import os
import re
import subprocess


if 'MODULEPATH' not in os.environ:
    f = open(os.environ['MODULESHOME'] + "/init/.modulespath", "r")
    path = []
    for line in f.readlines():
        line = re.sub("#.*$", '', line)
        if line is not '':
            path.append(line)
    os.environ['MODULEPATH'] = ':'.join(path)

if 'LOADEDMODULES' not in os.environ:
    os.environ['LOADEDMODULES'] = ''


def module(*args):
    if isinstance(args[0], list):
        args = args[0]
    else:
        args = list(args)
    (output, error) = subprocess.Popen(
            ['/usr/bin/modulecmd', 'python'] +
            args, stdout=subprocess.PIPE).communicate()
    exec(output)
