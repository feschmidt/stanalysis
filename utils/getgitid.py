"""Method for querying the current gitid

This module contains a single method used to query the current gitid of stlab.
This is necessary because stlab is constantly being developed and rewritten.
To be able to reproduce measurements, this module will save the git id together
with the measurement data.

"""

import os
import platform
import subprocess


def get_gitid(measfile):
    """Saves the gitid.

    Will query the gitid of stlabutils and save the id into a textfile in
    the same directory as the measurement data.

    Parameters
    ----------
    measfile : _io.TextIOWrapper
        File handle

    Returns
    -------
    gitid : str
        The current git id

    """
    theOS = platform.system()

    if theOS == 'Windows':
        cmd = 'git -C C:\\libs\\stlabutils rev-parse HEAD'
    elif theOS == 'Linux':
        cmd = 'git -C ~/git/stlabutils rev-parse HEAD'

    gitid = subprocess.check_output(cmd.split(' ')).decode("utf-8").strip('\n')

    filename = os.path.realpath(measfile.name)
    # dirname = dirname + '\\' + dirname
    with open(filename + '.stlabutils_id.txt', 'a') as myfile:
        myfile.write('# Current stlabutils gitid\n')
        myfile.write(gitid)
    print('Stlabutils git id:', gitid)
    return gitid
