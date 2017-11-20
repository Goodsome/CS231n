import numpy as np

a = np.random.randn(2,3)
c = np.array([1,2]).reshape(1,2)
b = a[[0,1],[[1,2]]].reshape(2,1)
print(a)
print(b.shape)
print(a - b)
