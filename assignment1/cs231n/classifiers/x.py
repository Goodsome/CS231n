import numpy as np

a = np.array([1,2,3,4,5])
b = np.array([2,2,2,2,2])

x = sum((b - a)**2)**0.5

print(x)
