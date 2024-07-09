import numpy as np
import cupy as cp

data = np.arange(10)  # Beispielarray
window_size = 3
offset = 2

# Umwandlung in ein CuPy-Array
data_cp = cp.asarray(data)

# Erstellen der überlappenden Fenster
batch_windows = cp.lib.stride_tricks.as_strided(data_cp, shape=(len(data) - window_size + 1, window_size),
                                                strides=(offset * data_cp.itemsize, data_cp.itemsize))

print(data_cp.itemsize)
print(batch_windows)