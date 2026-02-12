import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#%% Tests on the approximation for calibrationg delta_t

bar_c = 1
Cs = 2
theta = 1
sigma = 1

b = 3


Tmax = b / bar_c

Tmin = Tmax / 3

t = np.linspace(Tmin,Tmax, 200)

f = lambda t : (b - bar_c * t +  1/theta*(Cs - bar_c)*(1 - np.exp(-theta*t)) ) / ((sigma/theta)*np.sqrt(t - 2/theta *(1 - np.exp(-theta*t)) + 1/(2*theta)*(1 - np.exp(-2*theta*t))))

h = lambda t : (b )/(sigma * t * np.sqrt(t))
plt.figure(dpi = 200)
plt.xlabel(r"$\Delta_t$")
plt.plot(t, f(t))
plt.plot(t,h(t))
plt.show()

# epsilon = 0.001

# a = norm.isf(epsilon)
# print(a)

#%% Tests on simulation of \int_0^t C_s ds

def increment_C_intC(Y0, dt, barc, theta, sigma, Nt):
   
    # Definition of relevant constants 
    e = np.exp(- theta* dt)
    
    a = sigma**2 /(2*theta) * (1 -e**2)
    b = sigma**2 /(2 * theta**2) * (1 - e)**2
    c = sigma**2 /theta**2 * (dt -2/theta*(1 -e) + 1/(2*theta)*(1-e**2))
    
    v11 = np.sqrt(a)
    v21 = b/v11
    v22 = np.sqrt(c - v21**2)
    
    
    # Sampling
    Z = np.random.radn(Nt,2)
        


#%% Reaching desired size

