import numpy as np


def fft(x):
    """
    Compute the FFT of an array x using the Cooley-Tukey algorithm.
    """
    N = len(x)
    if N <= 1:
        return x

    # Recursive call for even and odd indices
    even = fft(x[0::2])
    odd = fft(x[1::2])

    # Combine the results
    T = []
    for k in range(N // 2):
        T.append(np.exp(-2j * np.pi * k / N) * odd[k])

    result = []
    for k in range(N // 2):
        result.append(even[k] + T[k])
    for k in range(N // 2):
        result.append(even[k] - T[k])

    return result