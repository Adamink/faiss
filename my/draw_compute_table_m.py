import numpy as np
import matplotlib.pyplot as plt
x = [1, 2, 4, 8, 16, 32]
y = [0.859, 0.928, 570.384, 621.805, 766.293, 1223.617]
plt.plot(x, y, label='Runtime of Computing Table')
plt.xlabel("number of subquantizers")
plt.ylabel("run time")
plt.title("Modeling PQ Table Computation")
plt.savefig("Modeling_PQ_Table_Computation.png")