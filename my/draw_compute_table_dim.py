import numpy as np
import matplotlib.pyplot as plt


with open("dim_result.txt", "r") as f:
    lines = f.readlines()
    x = []
    y = []
    for line in lines:
        dim, time = line.split(":")
        dim = int(dim)
        time = float(time)
        x.append(dim)
        y.append(time)

# // {sub q} x {(query id)(sub dim) * (code id)(sub dim)'} =>
# // {sub q} x {(query id)(code id)}
# m * nq *
plt.plot(x, y, label='Runtime of Computing Table(ms)')
plt.xlabel("Dims")
plt.ylabel("Run Time(ms)")
plt.title("Modeling PQ Table Computation")
plt.savefig("Modeling_PQ_Table_Computation_Dim.png")