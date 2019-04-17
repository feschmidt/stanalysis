"""Method for querying the current gitid

This module contains a single method used to query the current gitid of stlab and stlabutils.
This is necessary because the repositories are constantly being developed and rewritten.
To be able to reproduce measurements, this module will save the git ids together
with the measurement data.

"""

import os
import platform
import subprocess


def get_gitid(measfile):
    """Saves the gitid.

    Will query the gitid of both stlab and stlabutils and save the ids into a textfile in
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

    theids = []
    for repo in ['stlab', 'stlabutils']:
        if theOS == 'Windows':
            cmd = 'git -C C:\\libs\\' + repo + ' rev-parse HEAD'
        elif theOS == 'Linux':
            cmd = 'git -C ~/git/' + repo + ' rev-parse HEAD'

        gitid = subprocess.check_output(
            cmd.split(' ')).decode("utf-8").strip('\n')

        filename = os.path.realpath(measfile.name)
        # dirname = dirname + '\\' + dirname
        with open(filename + '.' + repo + '_id.txt', 'a') as myfile:
            myfile.write('# Current ' + repo + ' gitid\n')
            myfile.write(gitid)
        print(repo, 'git id:', gitid)
        theids.append(gitid)
    return theids
