import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

# calculates the electric field as a function of radius from the center of a charged ring.
x = np.linspace(0, 5, 200)
f = lambda a, theta: (a - np.cos(theta)) / (1 + a ** 2 - 2 * a * np.cos(theta))
y = [integrate.quad(lambda theta: f(xi, theta), 0, 2 * np.pi)[0] for xi in x]

plt.xlabel(r'$\frac{r}{R}$')
plt.ylabel(r'$\frac{E R}{k_e \lambda} $')
plt.title("Electric field inside a charged ring")
plt.plot(x, y)
plt.show()