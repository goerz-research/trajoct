import os
from subprocess import check_call


def get_pw(file):
    file = os.path.expanduser(file)
    with open(file) as in_fh:
        return in_fh.read().strip()


def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks"""
    if model['type'] != 'notebook':
        return  # only do this for notebooks
    d, fname = os.path.split(os_path)
    # check_call(['jupyter', 'nbconvert', '--to', 'script', fname], cwd=d)
    check_call(
        ['jupyter', 'nbconvert', '--to', 'html', '--output-dir=html', fname],
        cwd=d)


c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
c.NotebookApp.open_browser = False
c.NotebookApp.password = get_pw('~/.jupyter/password')
c.NotebookApp.port = 47962
c.FileContentsManager.post_save_hook = post_save
