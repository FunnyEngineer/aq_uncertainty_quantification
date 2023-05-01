import timeit

import numpy as np
import jax.numpy as jnp
from jax import random
def numpy_test():
    x = np.random.random([5000, 5000]).astype(np.float32)
    np.matmul(x, x)
    
def jax_test():
    x = random.uniform(random.PRNGKey(0), [5000, 5000])
    jnp.matmul(x, x)

print(timeit.timeit(numpy_test, number=1))
print(timeit.timeit(jax_test, number=1))