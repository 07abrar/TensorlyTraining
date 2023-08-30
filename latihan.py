import tensorly as tl
import numpy as np


tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)), dtype=tl.float64)
unfolded = tl.unfold(tensor, mode=0)
tl.fold(unfolded, mode=0, shape=tensor.shape)
