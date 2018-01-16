import numpy as np

x = np.arange(6).reshape(-1, 3)
x_mean = np.mean(x, axis=0)
x_var = np.var(x, axis=0)

x = (x - x_mean) / np.sqrt(x_var)
print(x)
