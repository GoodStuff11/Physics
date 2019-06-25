import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


L = 1
f = lambda x, y: np.sin(5 * x) * y ** 2
f_prime = lambda x, y: 0
dt = 0.001

elements = 50
dx = L / elements
X, Y = np.meshgrid(np.linspace(0, L, elements), np.linspace(0, L, elements), sparse=True)
Z = X * Y * (L - X) * (L - Y) * f(X, Y)

steps = 100
data = [Z]
data.append(np.zeros((elements, elements)))
# find first array of values then loop
for x in range(1, elements - 2):
    for y in range(1, elements - 2):
        u = data[0]
        data[1][x, y] = u[x, y] + dt * f_prime(x, y) + (dt / dx) ** 2 * (u[x + 1, y] + u[x, y + 1] - 4 * u[x, y] + u[x - 1, y] + u[x, y - 1])
for t in range(1, steps):
    for x in range(1, elements - 2):
        for y in range(1, elements - 2):
            u = data[t]
            data.append(np.zeros((elements, elements)))
            data[t + 1][x, y] = 2 * u[x, y] - data[t - 1][x, y] + (dt / dx) ** 2 * (u[x + 1, y] + u[x, y + 1] - 4 * u[x, y] + u[x - 1, y] + u[x, y - 1])

plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, data[60])
plt.show()
