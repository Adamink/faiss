import numpy as np
import matplotlib.pyplot as plt
x = list(range(0, 260, 25))
x[0] = 1
ob = [60.2, 54.7, 46.1, 34.1, 25.8, 27.3, 17.8, 14.6, 17.6, 16.0, 17.5]
pkb = [40.7, 17.7, 14.8, 10.7, 10.2, 10.3, 5.7, 5.5, 5.8, 5.4, 5.5]
plt.plot(x, ob, label='overall bandwidth(with matrixmult)')
plt.plot(x, pkb, label='kernel bandwidth')
plt.xlabel("k in k-select")
plt.ylabel("k-select bandwidth (GB/s)")
plt.title("Modelling K-select with dim=1")
plt.legend()
plt.savefig("Modelling_K-select_with_dim=1.png")