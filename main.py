import numpy as np

from fft import fft

# Example input
x = [10,1,5,17]

# Using our fft function
custom_fft = fft(x)

# Using numpy's fft function
numpy_fft = np.fft.fft(x)

# Comparison
print("X:", x)
print("Custom FFT:", custom_fft)
print("Numpy FFT:", numpy_fft)
print("Difference:", np.abs(custom_fft - numpy_fft))
