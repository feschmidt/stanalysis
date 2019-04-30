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

    try:
        filename = os.path.realpath(measfile.name)
    except AttributeError:
        filename = measfile
    with open(filename + '.gitids.txt', 'a') as myfile:
        for repo in ['stlab', 'stlabutils']:
            if theOS == 'Windows':
                cmd = 'git -C C:\\libs\\' + repo + ' rev-parse HEAD'
            elif theOS == 'Linux':  #or theOS == 'Darwin': # does not currently work for MacOS
                cmd = 'git -C ~/git/' + repo + ' rev-parse HEAD'
            else:
                print('Unknown platform detected:', theOS)
                break
            gitid = subprocess.check_output(
                cmd.split(' ')).decode("utf-8").strip('\n')

            myfile.write('# Current ' + repo + ' gitid\n')
            myfile.write(gitid + '\n')

            print(repo, 'git id:', gitid)
            theids.append(gitid)

    return theids


if __name__ == '__main__':
    get_gitid('.')