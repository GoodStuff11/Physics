import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def f():
    m = 1
    gamma = 1
    a = 1 / 4
    b = 1 / 2
    F0 = 1
    omega = 1

    def model(Y, t):
        # x1' = x2
        # m x2' = - gamma x2 + 2 a x1 - 4 b x1^3 + F0 cos(omega t)
        return [Y[1], -gamma / m * Y[1] + 2 * a * Y[0] - 4 * b * np.power(Y[0], 3) + F0 * np.cos(omega * t)]

    return model


t = np.linspace(0, 20, 1000)
y0 = 1
dydt0 = 0

for i in np.linspace(0, 2, 10):
    y = odeint(f(), [i, 0], t)
    plt.plot(t, [i[0] for i in y])
# add colour map
plt.xlabel('Time')
plt.ylabel('Position')
plt.show()
