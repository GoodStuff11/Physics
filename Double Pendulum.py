import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin
import matplotlib.animation as animation

from scipy.integrate import solve_ivp

# angle in units of degrees
try:
    theta0, theta1, m1, m2, l1, l2 = [float(x) for x in sys.argv[1:]]
except ValueError:
    sys.exit("Command line argument(s) cannot be converted to floats")

# convert to radians
theta0 = pi * theta0 / 180
theta1 = pi * theta1 / 180
g = 1


def solve(t, z):
    # theta1 = z[0]
    # theta2 = z[1]
    # theta1' = z[2]
    # theta2' = z[3]
    alpha1 = l2 / l1 * m2 / (m1 + m2) * cos(z[0] - z[1])
    alpha2 = l1 / l2 * cos(z[0] - z[1])
    f1 = -l2 / l1 * m2 / (m1 + m2) * z[3] ** 2 * sin(z[0] - z[1]) - g / l1 * sin(z[0])
    f2 = l1 / l2 * z[2] ** 2 * sin(z[0] - z[1]) - g / l2 * sin(z[1])
    return [z[2], z[3], (f1 - alpha1 * f2) / (1 - alpha1 * alpha2), (-alpha2 * f1 + f2) / (1 - alpha1 * alpha2)]


sol = solve_ivp(solve, (0, 30), [theta0, theta1, 0, 0], t_eval=np.linspace(0, 30, 1000))


def update_line(frame, line, scatter, path):
    x = [0, l1 * sin(sol.y[0][frame]), l1 * sin(sol.y[0][frame]) + l2 * sin(sol.y[1][frame])]
    y = [0, -l1 * cos(sol.y[0][frame]), - l1 * cos(sol.y[0][frame]) - l2 * cos(sol.y[1][frame])]
    line.set_data(x, y)
    scatter.set_data(x[1:], y[1:])
    if frame != 0:
        path.set_data(list(path.get_xdata()) + [x[-1]], list(path.get_ydata()) + [y[-1]])
    else:
        path.set_data([x[-1]], [y[-1]])
    return line, scatter, path


plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\bin\\ffmpeg.exe'
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)


fig = plt.figure("Double pendulum")
scale = 4
plt.xlim([-scale, scale])
plt.ylim([-scale, scale])
l, = plt.plot([], [], 'r')
s, = plt.plot([], [], 'ko')
path, = plt.plot([], [], 'b')
path.set_linewidth(0.5)

plt.axis('off')
pendulum = animation.FuncAnimation(fig, update_line, 1000, fargs=(l, s, path), interval=1, blit=True)
pendulum.save('double pendulum.mp4', writer=writer)

plt.figure("Double pendulum angle plot")
plt.plot(sol.t, sol.y[0], color='r', label=r'$\theta_1$')
plt.plot(sol.t, sol.y[1], color='b', label=r'$\theta_2$')
plt.ylabel("Angle (radians)")
plt.xlabel("Time")
plt.legend()
plt.grid()

plt.figure("Energy")
KE = 0.5 * m1 * sol.y[2] ** 2 * l1 ** 2 + 0.5 * m2 * (l1 ** 2 * sol.y[2] ** 2 + l2 ** 2 * sol.y[3] ** 2 + 2 * sol.y[2] * l1 * sol.y[3] * l2 * cos(sol.y[0] - sol.y[1]))
PE = -m1 * l1 * cos(sol.y[0]) - m2 * (l1 * cos(sol.y[0]) + l2 * cos(sol.y[1]))
plt.plot(sol.t, KE, color='r', label='Kinetic Energy')
plt.plot(sol.t, PE, color='b', label='Potential Energy')
plt.plot(sol.t, KE + PE, color='k', label='Total Mech')
plt.legend()
plt.grid()

plt.show()

# frame = 100
# plt.figure("Double pendulum")
# scale = 4
# plt.xlim([-scale, scale])
# plt.ylim([-scale, scale])
# plt.axis('off')
# x = [0, l1 * sin(sol.y[0][frame]), l1 * sin(sol.y[0][frame]) + l2 * sin(sol.y[1][frame])]
# y = [0, -l1 * cos(sol.y[0][frame]), - l1 * cos(sol.y[0][frame]) - l2 * cos(sol.y[1][frame])]
# plt.plot(x, y, color='r')
# plt.scatter(x[1:], y[1:], color='k', marker='o')
# plt.show()
