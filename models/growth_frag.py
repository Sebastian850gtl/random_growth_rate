import numpy as np
import heapq

class Cell():
    """ Encodes all important informations about a cell """
    def __init__(self, birth_time, birth_size):
        self.birth_time = birth_time
        self.birth_size = birth_size

        self.division_size = None
        self.division_time = None

    def get_value(self):
        return (self.birth_time, self.division_time, self.birth_size, self.division_size)
    

    def sample_division_size_parametrized(self, alpha , offset ):
        r""" Samples first hitting time with K(x) = 1_{x \ge offset} * (x - offset)^alpha"""
        x0 = self.birth_size
        U = np.random.rand()
        if x0 <= offset:
            return offset + ((1 + alpha) * -np.log(U) )**(1 / (1 + alpha))
        else:
            return offset + ((1 + alpha) * -np.log(U) + (x0 - offset)**(1+alpha))**(1 / (1 + alpha))
    
    def get_time(self, growth_rate):
        bt, lt, bs, ds = self.get_value()
        if ds == None:
            raise ValueError('The division size has not been computed yet')
        
        return 1/growth_rate * np.log(ds / bs)

    def division(self):
        """ Divides a cell into two new daughters with the rule of equal mitosis """
        bt, dt, bs, ds = self.get_value()
        if ds == None:
            raise ValueError('The division size has not been computed yet')
        elif dt == None:
            raise ValueError('The life time has not been computed yet')
        else:
            c1 = Cell( dt, ds/2)
            c2 = Cell( dt, ds/2)
        return c1, c2

class GF_equal_mitosis():
    """ Object model of growth fragmentation with exponential growth
    Allows to simulate single samples of this model"""
    def __init__(self,growth_rate,division_rate_params):
        """
        Arguments:

        growth_rate : positive, float; growth_rate in the exponential growth
        division_rate : function, division_rate(x) is the probability that a polymer of size x divides.
        division_rate_params = (alpha, offset) : params of the parametrized K
        """

        self.growth_rate = growth_rate
        #self.K = f_division_rate
        self.div_params = division_rate_params

    def run(self, Tmax, init, itemax = 1e6):
        offset, alpha = self.div_params
        growth_rate = self.growth_rate

        # Compute for the first cell
        init.division_size = init.sample_division_size_parametrized(alpha, offset)
        init.division_time = init.get_time(growth_rate)   

        # Initialize loop parameters
        init_key = '0'
        times_and_keys = [(init.division_time, init_key)]     
        all_cells = {init_key:init}

        counter = 0
        heapq.heapify(times_and_keys)
        while times_and_keys[0][0] < Tmax and counter < itemax:
            _,mother_key = heapq.heappop(times_and_keys)
            mother = all_cells[mother_key]
            
            offspring1, offspring2 = mother.division()

            for id_ofsp, ofsp in enumerate((offspring1, offspring2)):
                counter += 1

                ofsp_key = str(counter)
                all_cells[ofsp_key] = ofsp
                
                ofsp.division_size = ofsp.sample_division_size_parametrized(alpha, offset)
                ofsp.division_time = ofsp.get_time(growth_rate) + ofsp.birth_time
                
                heapq.heappush(times_and_keys, (ofsp.division_time, ofsp_key))

        return all_cells, times_and_keys
            

        



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import bisect
    import heapq

    
    growth_rate = 1
    division_rate_params = offset, alpha = 0, 0
    Tmax = 1
    init = Cell(0,1)
    Model = GF_equal_mitosis(growth_rate = growth_rate, division_rate_params = division_rate_params)
    res, times = Model.run(Tmax, init)

    for cell_key in res:
        print(res[cell_key].get_value())