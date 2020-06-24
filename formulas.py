import numpy as np

# A collection of generally used formulas for microwave engineering


def SNR(S, N, mode='power'):
    # Signal to noise ratio
    if mode == 'power':
        return S / N
    elif mode == 'dB':
        return dBtoP(S, ref=1) / dBtoP(N, ref=1)
    elif mode == 'dBm':
        return dBtoP(S, ref=1e-3) / dBtoP(N, ref=1e-3)
    else:
        return KeyError('mode unknown')


def dBtoP(dB, ref=1e-3):
    # convert dBref to linear, depending on reference
    # ref=1 for dB, ref=1e-3 for dBm
    return 10**(dB / 10) * ref


def PtodB(P, ref=1e-3):
    # convert linear to dBref, depending on reference
    # ref=1 for dB, ref=1e-3 for dBm
    return 10 * np.log10(P / ref)
