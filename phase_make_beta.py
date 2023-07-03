import numpy as np

b0 = 1.0
p = 1.4
n = 25

res = [b0 * p**i for i in range(n)]
print(",".join(map(str, res)))