import numpy as np
import heapq
import matplotlib.pyplot as plt
import bisect

from models.random_gf import GF_equal_mitosis, Cell

def alive_cells_at_t(t, all_cells, birth_times, division_times):
    # index of first cell born after t
    idx_birth = bisect.bisect_right(birth_times, t)

    # alive if division_time >= t
    alive = [all_cells[i] for i in range(idx_birth)
             if division_times[i] >= t]
    return alive

# Case 1 
#np.random.seed(13)

growth_rate = 1
theta = 1
sigma = 1
init_birth_size = 1

ou_growth_rate_parameters = (growth_rate, theta, sigma)
division_rate_params = offset, alpha = 0, 0

dtmin = 1e-5
eps = 0.01

Tmax = 13
itemax = 1e7

Tmin = Tmax / 2
time_points = np.linspace(Tmin, Tmax, 19)

init = Cell(0,init_birth_size, growth_rate)
Model = GF_equal_mitosis(ou_growth_rate_parameters = ou_growth_rate_parameters, division_rate_params = division_rate_params)
all_cells, times = Model.run(Tmax, init, dtmin = dtmin, eps = eps)

birth_times = [c.birth_time for c in all_cells.values()]
division_times = [c.division_time for c in all_cells.values()]

mean_sizes = []
number_of_cells = []
for t in time_points:
    alive_cells = alive_cells_at_t(t, all_cells, birth_times, division_times)
    cell_sizes_at_t = [c.birth_size* np.exp(growth_rate* (t - c.birth_time)) for c in alive_cells]

    n = len(alive_cells)
    print("Number of cells alive at t = %.2f: %4d "%(t, n))
    first_moment = np.sum(cell_sizes_at_t)
    number_of_cells.append(n)
    mean_sizes.append(first_moment)

number_of_cells = np.array(number_of_cells)

# log_coeff = (np.log(number_of_cells[1:]) - np.log(number_of_cells[:-1]))/(time_points[1:] - time_points[:-1])
# print("Mean coeff is :%.5f"%(np.mean(log_coeff)))
# print(len(log_coeff))

# plt.figure(dpi = 200)
# plt.plot(time_points, number_of_cells)
# plt.plot(time_points, np.exp((growth_rate )*time_points))
# plt.show()

plt.figure(dpi = 200)
plt.plot(time_points, np.log(number_of_cells) - np.log(number_of_cells)[0] + Tmin*growth_rate)
plt.plot(time_points, growth_rate * time_points)
plt.show()
# plt.figure(dpi = 200)
# plt.plot(time_points[1:], log_coeff)
# plt.plot(time_points, growth_rate * np.ones(len(time_points)))
# plt.show()



