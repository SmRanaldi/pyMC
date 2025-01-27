import numpy as np
import envelope as env
import envelope_py as envpy

from timeit import timeit

mod = np.concatenate([
    np.zeros((1,1000)).flatten()+0.01,
    np.ones((1,1000)).flatten(),
    np.zeros((1,1000)).flatten()+0.01,
], axis=0)

emg = np.random.randn(1,3000)*mod
emg = emg.transpose()
# emg = emg.reshape(-1,1)

print(f"Python: {timeit(lambda: envpy.adaptive_envelope(emg[:,0]), number=1):.4e}")
print(f"Cython: {timeit(lambda: env.adaptive_envelope(emg[:,0]), number=1):.4e}")