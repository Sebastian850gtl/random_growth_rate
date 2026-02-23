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

def increment_C_intC(C0, dt, barc, theta, sigma, Nt):
    res = np.empty((Nt,2))
    
    # Definition of relevant constants 
    e = np.exp(- theta* dt)
    
    # Constant interecept
    intercept = np.array(( 1 - e, dt  + (e - 1)/theta)) * barc
    
    # Linear update
    lin = np.array( ((e , 0 ), ((1 - e)/theta, 1)) )
    
    #Noise, Cholesky matrix
    a = sigma**2 /(2*theta) * (1 -e**2)
    b = sigma**2 /(2 * theta**2) * (1 - e)**2
    c = sigma**2 /theta**2 * (dt -2/theta*(1 -e) + 1/(2*theta)*(1-e**2))
    
    v11 = np.sqrt(a)
    v21 = b/v11
    v22 = np.sqrt(c - v21**2)
    Cholesky = np.array(( (v11,0),(v21,v22)))
    
    
    # Sampling
    Z = np.random.randn(Nt-1,2)
    dBt = Z.dot(Cholesky.T)
    
    #Loop
    
    Yk = np.array((C0, 0))
    res[0,:] = Yk
    for k in range(Nt-1):
        Yk = intercept + Yk.dot(lin.T) + dBt[k]
        res[k+1] = Yk
    return res

if __name__ == "__main__":
    Nt = 20
    dt = 0.05
    sigma = 1
    theta = 1
    barc = 1
    C0 = 1
    
    res = increment_C_intC(C0, dt, barc, theta, sigma, Nt)
    
    times = np.linspace(0, (Nt - 1)*dt, Nt)
    plt.figure(dpi = 200)
    plt.title("C_t")
    plt.plot(times,res[:,0])
    
    plt.figure(dpi = 200)
    plt.title("intC_t")
    plt.plot(times,res[:,1])
    
    # Checking mean
    
    
    Nsamples = 10000
    mean = 0
    for k in range(Nsamples):
        res = increment_C_intC(C0, dt, barc, theta, sigma, Nt)
        mean += res[:,1]
    mean = mean / Nsamples
    
    
    var = 0
    for k in range(Nsamples):
        res = increment_C_intC(C0, dt, barc, theta, sigma, Nt)
        var += (res[:,1] - mean)**2
    var = var / (Nsamples - 1)
    
    
    f = lambda t : barc * t
    plt.figure(dpi = 200)
    plt.title("Test mean")
    plt.plot(times,mean)
    plt.plot(times,f(times))
    
    g = lambda t : sigma**2 / theta**2 * ( t - 2/theta * (1 - np.exp(-theta*t)) + 1/(2*theta)*(1 - np.exp(-2*theta * t)))
    plt.figure(dpi = 200)
    plt.title("Test variance")
    plt.plot(times,var)
    plt.plot(times,g(times))
        

#%% Reaching desired size

