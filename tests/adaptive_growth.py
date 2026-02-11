import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

bar_c = 1
Cs = 2
theta = 1
sigma = 1

b = 3


Tmax = b / bar_c

Tmin = Tmax / 100

t = np.linspace(Tmin,Tmax, 200)

f = lambda t : (b - bar_c * t +  1/theta*(Cs - bar_c)*(1 - np.exp(-theta*t)) ) / ((sigma/theta)*np.sqrt(t - 2/theta *(1 - np.exp(-theta*t)) + 1/(2*theta)*(1 - np.exp(-2*theta*t))))

# g = lambda t : - bar_c * theta / sigma * np.sqrt(t)

h = lambda t : (b )/(sigma * t * np.sqrt(t))
plt.plot(t, f(t))
# plt.plot(t,g(t))
plt.plot(t,h(t))
plt.show()

# epsilon = 0.001

# a = norm.isf(epsilon)
# print(a)
