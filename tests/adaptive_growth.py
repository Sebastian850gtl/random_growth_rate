import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import optimize
from scipy.optimize import newton
from scipy.optimize import bisect

#%% Tests on the approximation for calibrationg delta_t

# bar_c = 1
# Cs = -5
# theta = 1
# sigma = 1

# b = 3


# Tmax = 10

# Tmin = Tmax / 10

# t = np.linspace(Tmin,Tmax, 200)

# f = lambda t : (b - bar_c * t -  1/theta*(Cs - bar_c)*(1 - np.exp(-theta*t)) ) / ((sigma/theta)*np.sqrt(t - 2/theta *(1 - np.exp(-theta*t)) + 1/(2*theta)*(1 - np.exp(-2*theta*t))))

# h = lambda t : (b - Cs*t)/(sigma * t ) #* np.sqrt(t))
# # print(f(Tmax))
# plt.figure(dpi = 200)
# plt.xlabel(r"$\Delta_t$")
# plt.plot(t, norm.sf(f(t)))
# plt.plot(t, norm.sf(h(t)))
# plt.show()
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

def single_increment(Yk, t, barc, theta, sigma):
    # Definition of relevant constants 
    e = np.exp(- theta* t)
    
    # Constant interecept
    intercept = np.array(( 1 - e, t  + (e - 1)/theta)) * barc
    
    # Linear update
    lin = np.array( ((e , 0 ), ((1 - e)/theta, 1)) )
    
    #Noise, Cholesky matrix
    if theta *t < 1e-3: # First order approximation, avoids negative values because of floating point
        a = sigma**2 * t
        b = sigma**2 /2  * t **2
        c = sigma**2 /3 * t**3
    else:
        a = sigma**2 /(2*theta) * (1 -e**2)
        b = sigma**2 /(2 * theta**2) * (1 - e)**2
        c = sigma**2 /theta**2 * (t -2/theta*(1 -e) + 1/(2*theta)*(1-e**2))
    
    v11 = np.sqrt(a)
    v21 = b/v11
    v22 = np.sqrt(c - v21**2)
    Cholesky = np.array(( (v11,0),(v21,v22)))
    
    
    # Sampling
    Z = np.random.randn(2)
    dBt = Z.dot(Cholesky.T)
    
    return intercept + Yk.dot(lin.T) + dBt

def root_function(t, Ck, barc, theta, sigma, tol, remaining):
    e = np.exp(-theta * t)
    mt = barc * t + (Ck - barc)/theta * (1 - e)

    if theta * t < 1e-4:
        sigmat = sigma * np.sqrt(t**3 / 3)
    else:
        A = t - 2/theta*(1 - e) + 1/(2*theta)*(1 - e**2)
        sigmat = sigma/theta * np.sqrt(A)

    return mt + sigmat * tol - remaining

def sample_time_adaptive_method(tresh , eps, dtmin, C0, barc, theta, sigma):
    r""" 

    Y is the vector (C_t, \int_0^t C_s ds ) 

    """
    tol = norm.isf(eps)

    dt =  (tresh * np.sqrt(3)/ (sigma * tol)  )**(2/3)
    
    intC = 0
    Y = np.array((C0, intC)) # Y is the vector (C_t, \int_0^t C_s ds ) 
    t = 0
    Ck = C0
    
    # For test
    times = [0]
    values = [0]
    #
    counter_ite = 0
    while intC < tresh:
        counter_ite += 1
        remaining = tresh -intC

        if dt < dtmin:
            dt = dtmin
        elif root_function(dt, Ck, barc, theta, sigma, tol, remaining) > 0:
            dt = optimize.bisect(root_function, 0, dt, xtol = dtmin, args = (Ck, barc, theta, sigma, tol, remaining))
            dt = max(dt,dtmin)
        else:
            pass
        
        t = t + dt

        Y = single_increment(Y, dt, barc, theta, sigma)
        Ck, intC = Y

        #For tests
        times.append(t)
        values.append(intC)
        #
    
    return t  + (tresh - intC)/Ck, t, values, times


if __name__ == "__main__":

#%% Reaching desired size
    # time_estimation1, time_estimation2, values, times = sample_time_adaptive_method(tresh , eps, dtmin, C0, barc, theta, sigma)
    # number_of_iterations = len(values) - 1
    # print(time_estimation1, time_estimation2, number_of_iterations)

    # plt.figure(dpi = 200)
    # plt.scatter(times, values, color = 'darkorange', marker = "*", label = r"values of $\int_0^t C_s \mathrm{d}s$")
    # plt.plot(times, tresh* np.ones(number_of_iterations + 1), label = "treshold", linestyle = "--")
    # plt.legend()
    # plt.show()

    # sigma = 1
    # theta = 1
    # barc = 1
    
    # C0 = 1

    # tresh = 1
    # dtmin = 1e-8
    # eps = 1e-6

    # N_samples = 3*int(1e4)
    # time_samples1, time_samples2, iteration_numbers = [], [], []

    # weird_values, weird_times, weird_iteration_number = [], [], 0
    # for k in range(N_samples):
    #     np.random.seed(k)

    #     print("Iteration number", k)
    #     time_estimation1, time_estimation2, values, times = sample_time_adaptive_method(tresh , eps, dtmin, C0, barc, theta, sigma)
    #     number_of_iterations = len(values) - 1

    #     if time_estimation1 > 13:
    #         weird_values, weird_times, weird_iteration_number = values, times, number_of_iterations
        
    #     time_samples1.append(time_estimation1)
    #     time_samples2.append(time_estimation2)
    #     iteration_numbers.append(number_of_iterations)
    



    # if weird_iteration_number > 0:
    #     print(weird_iteration_number)
    #     plt.figure(dpi = 200)
    #     plt.scatter(weird_times, weird_values, color = 'darkorange', marker = "*", label = r"values of $\int_0^t C_s \mathrm{d}s$")
    #     plt.plot(weird_times, tresh* np.ones(weird_iteration_number + 1), label = "treshold", linestyle = "--")
    #     plt.legend()
    #     plt.show()
    # else:
    #     pass

    # print(np.max(time_samples1))
    # plt.figure(dpi = 200)
    # plt.title("Hitting time")
    # plt.hist(time_samples1, bins = N_samples//100)
    # plt.show()

    # plt.figure(dpi=200)
    # plt.title("Number of iterations (log scale)")
    # plt.boxplot(iteration_numbers)
    # plt.yscale('log')
    # plt.show()

    def sampling(tresh , eps, dtmin, C0, barc, theta, sigma, N_samples):
        time_samples1, time_samples2, iteration_numbers = [], [], []

        for k in range(N_samples):
            np.random.seed(k)

            print("Iteration number", k)
            time_estimation1, time_estimation2, values, times = sample_time_adaptive_method(tresh , eps, dtmin, C0, barc, theta, sigma)
            number_of_iterations = len(values) - 1

            
            time_samples1.append(time_estimation1)
            time_samples2.append(time_estimation2)
            iteration_numbers.append(number_of_iterations)
        return time_samples1, iteration_numbers

    eps_list = [ 1e-5, 1e-6, 1e-10]
    N_samples = int(1e4)
    dtmin = 1e-6

    C0 = 1
    barc = 1
    theta = 1
    sigma = np.sqrt(2*theta)
    tresh = 1

    mean_times = []
    for eps in eps_list:
        
        res = sampling(tresh , eps, dtmin, C0, barc, theta, sigma, N_samples)

        time_samples1, iteration_numbers = res
        mean_times.append(np.mean(time_samples1))

    
    print(mean_times)

    # # Theoretical functions
    # f = lambda t: barc * t + (C0 - barc)/theta * (1 - np.exp(-theta*t))
    # g = lambda t: sigma**2 / theta**2 * ( t - 2/theta * (1 - np.exp(-theta*t)) + 1/(2*theta)*(1 - np.exp(-2*theta * t)) )

    # f1 = lambda t: barc + (C0 - barc)* np.exp(-theta*t)

    # import numpy as np
    # import matplotlib.pyplot as plt

    # # Parameters
    # Nt = 1000
    # T = 2000
    # dt = T / Nt
    # sigma = 1
    # theta = 0.5
    # barc = 1
    # C0 = 1
    # times = np.linspace(0, (Nt - 1)*dt, Nt)

    # # Plot multiple realizations in a square (2x4)
    # fig, axs = plt.subplots(2, 2, figsize=(16, 8), dpi=200)

    # for i in range(2):
    #     for j in range(1):
    #         res = increment_C_intC(C0, dt, barc, theta, sigma, Nt)
        
    #         row = i 
    #         col = 2*j
        
    #         # C_t
    #         axs[row, col].plot(times, res[:,0])
    #         axs[row, col].set_title(r"$t \mapsto C_t$")
    #         axs[row, col].plot(times, f1(times), linestyle = "--", label = 'theoretical mean')
    #         axs[row, col].legend(fontsize=6)

    #         # intC_t
    #         axs[row, col + 1].plot(times, res[:,1])
    #         axs[row, col + 1].plot(times, f(times), linestyle = "--", label = 'theoretical mean')

    #         axs[row, col + 1].set_title(r"Plot of $t \mapsto \int_0^t C_s \mathrm{d}s$")
    #         axs[row, col+1].legend(fontsize=6)

    # plt.tight_layout()
    # plt.show()

    # # Parameters
    # Nt = 100
    # T = 10
    # dt = T / Nt
    # sigma = 1
    # theta = 1
    # barc = 1
    # C0 = 7
    # times = np.linspace(0, (Nt - 1)*dt, Nt)
    # # Compute mean and variance 
    # Nsamples = 10000
    # mean = np.zeros(Nt)
    # var = np.zeros(Nt)

    # for k in range(Nsamples):
    #     res = increment_C_intC(C0, dt, barc, theta, sigma, Nt)
    #     mean += res[:,1]
    # mean /= Nsamples

    # for k in range(Nsamples):
    #     res = increment_C_intC(C0, dt, barc, theta, sigma, Nt)
    #     var += (res[:,1] - mean)**2
    # var /= (Nsamples - 1)


    # # Plot test mean
    # plt.figure(dpi=200)
    # plt.plot(times, mean, label="Simulated mean")
    # plt.plot(times, f(times), label="Theoretical mean", linestyle="--")
    # plt.title("Test mean")
    # plt.legend()
    # plt.show()

    # # Plot test variance
    # plt.figure(dpi=200)
    # plt.plot(times, var, label="Simulated variance")
    # plt.plot(times, g(times), label="Theoretical variance", linestyle="--")
    # plt.title("Test variance")
    # plt.legend()
    # plt.show()
