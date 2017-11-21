import numpy as np

X = np.random.randn(5,3)
i = np.random.choice(5,3,replace = False)
x = X[i]
print(i)
print(X)
print(x)
