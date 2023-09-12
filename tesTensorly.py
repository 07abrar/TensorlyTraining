import tensorly as tl
import numpy as np

tl.set_backend('pytorch') # Or 'mxnet', 'numpy', 'tensorflow', 'cupy' or 'jax'
tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)))
type(tensor) # torch.Tensor

from tensorly.decomposition import tucker
# Apply Tucker decomposition
tucker_tensor = tucker(tensor, rank=[2, 2, 2])
# Reconstruct the full tensor from the decomposed form
tl.tucker_to_tensor(tucker_tensor)
