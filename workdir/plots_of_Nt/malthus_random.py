import numpy as np
import heapq
import matplotlib.pyplot as plt
import bisect
from time import time

from models.random_gf import GF_equal_mitosis, Cell

def alive_cells_at_t(t, all_cells, birth_times, division_times):
    # index of first cell born after t
    idx_birth = bisect.bisect_right(birth_times, t)

    # alive if division_time >= t
    alive = [all_cells[i] for i in range(idx_birth)
             if division_times[i] >= t]
    return alive

# Case 1 
#np.random.seed(0)

growth_rate = 1
theta = 0.01
sigma = np.sqrt(2*theta)

init_birth_size = 1

ou_growth_rate_parameters = (growth_rate, theta, sigma)
division_rate_params = offset, alpha = 1, 2

eps = 1e-8
dtmin = 1e-6

number_of_time_points = 20
max_number_of_cells = 1e5

N_samples = 6

plt.figure(dpi = 200)

Tmax_max = 0
Tmin_min = 0
for id_sample in range(N_samples):
    init = Cell(0,init_birth_size, growth_rate)

    print("Starting simulation id :",id_sample)
    t0 = time()
    Model = GF_equal_mitosis(ou_growth_rate_parameters = ou_growth_rate_parameters, division_rate_params = division_rate_params)
    all_cells, times = Model.run(Tmax = np.inf, init = init, dtmin = dtmin, eps = eps, max_number_of_cells = max_number_of_cells)
    print("...done in %.3f s"%(time() - t0))

    Tmax = times[0][0]
    Tmin = 0.95 * Tmax 
    time_points = np.linspace(Tmin, Tmax, number_of_time_points)

    birth_times = [c.birth_time for c in all_cells.values()]
    division_times = [c.division_time for c in all_cells.values()]

    mean_sizes = []
    number_of_cells = []

    #print("Starting computation of time slices")
    #t0 = time()
    for t in time_points:
        alive_cells = alive_cells_at_t(t, all_cells, birth_times, division_times)
        cell_sizes_at_t = [c.birth_size* np.exp(growth_rate* (t - c.birth_time)) for c in alive_cells]

        n = len(alive_cells)
        print("Number of cells alive at t = %.2f: %4d "%(t, n))
        first_moment = np.sum(cell_sizes_at_t)
        number_of_cells.append(n)
        mean_sizes.append(first_moment)
    #print("...done in %.3f s"%(time() - t0))

    number_of_cells = np.array(number_of_cells)
    plt.plot((time_points - Tmin)/(Tmax - Tmin), (np.log(number_of_cells) - np.log(number_of_cells)[0])/(Tmax - Tmin) )
    
    #plt.plot(time_points - Tmin, np.log(number_of_cells) /time_points)
    #plt.plot(time_points, number_of_cells)
    
    Tmax_max = max (Tmax - Tmin, Tmax_max)

time_points = np.linspace(0, 1, number_of_time_points)
plt.plot(time_points, growth_rate * time_points, color = 'blue', linestyle = 'dashdot')
#plt.plot(time_points, growth_rate * np.ones([len(time_points)]), color = 'blue', linestyle = 'dashdot')
plt.show()



