from time import perf_counter_ns
import numpy as np
import jax.numpy as jnp
from jax import jit

def time_func(func, name):
    start = perf_counter_ns()
    res = func()
    end = perf_counter_ns()
    dur = end - start
    print(f"{name}() result = {res}")
    print(f"duration: {dur}ns ({dur/1_000:.0}µs) ({dur/1_000_000:.0}ms) ({dur/1_000_000_000:.0}sec)")

data = np.random.normal(size=(1_000, 1))

def basic_python():
    result = 0
    for i in range(0, 1_000):
        result += data.T[0, i] * data[i, 0]
    return result

def numpy():
    return data.T @ data

def jax_no_jit():
    return data.T @ data

@jit
def mm(data):
    return data.T @ data
def jax_jit():
    return mm(data)

time_func(basic_python, "basic_python")
time_func(numpy, "numpy")
time_func(jax_no_jit, "jax_no_jit")
time_func(jax_jit, "jax_jit")
time_func(jax_jit, "jax_jit_again")