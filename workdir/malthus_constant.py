import numpy as np
import heapq
import matplotlib.pyplot as plt
import bisect

from models.growth_frag import GF_equal_mitosis, Cell

def alive_cells_at_t(t, all_cells, birth_times, division_times):
    # index of first cell born after t
    idx_birth = bisect.bisect_right(birth_times, t)

    # alive if division_time >= t
    alive = [all_cells[str(i)] for i in range(idx_birth)
             if division_times[i] >= t]
    return alive

# Case 1 

growth_rate = 1
division_rate_params = offset, alpha = 0, 0


Tmax = 1

time_points = np.linspace(0, Tmax, 40)
init = Cell(0,1)
Model = GF_equal_mitosis(growth_rate = growth_rate, division_rate_params = division_rate_params)
all_cells, times = Model.run(Tmax, init)

birth_times = [c.birth_time for c in all_cells.values()]
division_times = [c.division_time for c in all_cells.values()]

mean_sizes = []
for t in time_points:
    alive_cells = alive_cells_at_t(t, all_cells, birth_times, division_times)
    cell_sizes_at_t = [c.birth_size* np.exp(growth_rate* (t - c.birth_time)) for c in alive_cells]

    print("Number of cells alive at t = %.2f: %4d "%(t, len(alive_cells)))
    first_moment = np.sum(cell_sizes_at_t)
    mean_sizes.append(first_moment)

print(mean_sizes)
plt.figure(dpi = 200)
plt.plot(time_points, np.log(np.array(mean_sizes)))
plt.plot(time_points, growth_rate * time_points)
plt.show()


