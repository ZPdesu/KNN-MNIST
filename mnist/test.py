import numpy as np
from scipy.stats import mode

a = np.array([2,8,3,2,2,6])
u = a.argsort()
z = mode(a)
print z[0][0]

print z[0]