import numpy as np
import matplotlib.pyplot as plt
x = list(range(100, 1050, 100))
t1 = 3.312
y = [3.428, 3.465, 4.627, 4.614, 4.565, 4.789, 5.281, 5.395, 5.42, 5.492]
y1 = [_ - t1 for _ in y]
plt.plot(x, y1)
plt.xlabel("dim")
plt.ylabel("t2 - t1(ms)")
plt.title("Modelling Scan")
plt.savefig("modeling_scan.png")