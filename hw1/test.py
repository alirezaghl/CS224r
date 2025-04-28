import numpy as np

batch_size = 128
minibatch = np.random.permutation(batch_size)

print(minibatch.shape)
print(batch_size.shape)