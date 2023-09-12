import numpy as np
from timeit import default_timer as timer
from numba import cuda, jit

#CPU
def FillArrayWithoutGPU(a):
    for k in range(1000000):
        a[k]+=1

#GPU
@jit(target_backend='cuda')
def FillArrayWithGPU(a):
    for k in range(1000000):
        a[k]+=1

a = np.ones(1000000, dtype = np.float64)

start = timer()
FillArrayWithoutGPU(a)
print("On a CPU: ",timer()-start)

start = timer()
FillArrayWithGPU(a)
print("On a GPU: ",timer()-start)
