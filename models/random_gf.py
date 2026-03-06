import numpy as np
from scipy.stats import norm
from scipy import optimize
import heapq

def single_increment(Yk, t, barc, theta, sigma):
    # Definition of relevant constants 
    e = np.exp(- theta* t)
    
    # Constant interecept
    intercept = np.array(( 1 - e, t  + (e - 1)/theta)) * barc
    
    # Linear update
    lin = np.array( ((e , 0 ), ((1 - e)/theta, 1)) )
    
    #Noise, Cholesky matrix
    if theta *t < 1e-4: # First order approximation, avoids negative values because of floating point
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
    dt = (sigma * tol / (tresh*np.sqrt(3)))**(-2/3)
    intC = 0
    Y = np.array((C0, intC)) # Y is the vector (C_t, \int_0^t C_s ds ) 
    t = 0
    Ck = C0

    counter_ite = 0
    while intC < tresh:
        counter_ite += 1
        remaining = tresh -intC

        if dt < dtmin:
            dt = dtmin
        elif root_function(dt, Ck, barc, theta, sigma, tol, remaining) > 0:
            dt = optimize.bisect(root_function, 0, dt, xtol = dtmin, args = (Ck, barc, theta, sigma, tol, remaining))
        else:
            pass
        t = t + dt

        Y = single_increment(Y, dt, barc, theta, sigma)
        Ck, intC = Y
    return t - dt + tresh/Ck , Ck

class Cell():
    """ Encodes all important informations about a cell """
    def __init__(self, birth_time, birth_size, birth_growth_rate):
        self.birth_time = birth_time
        self.birth_size = birth_size
        self.birth_growth_rate = birth_growth_rate

        self.division_size = None
        self.division_time = None
        self.division_growth_rate = None

    def get_value(self):
        return (self.birth_time, self.division_time, self.birth_size, self.division_size,self.birth_growth_rate, self.division_growth_rate)
    

    def sample_division_size_parametrized(self, alpha , offset ):
        r""" Samples first hitting time with K(x) = 1_{x \ge offset} * (x - offset)^alpha"""
        x0 = self.birth_size
        U = np.random.rand()
        if x0 <= offset:
            return offset + ((1 + alpha) * -np.log(U) )**(1 / (1 + alpha))
        else:
            return offset + ((1 + alpha) * -np.log(U) + (x0 - offset)**(1+alpha))**(1 / (1 + alpha))
    
    def get_time(self, dtmin, eps, ou_growth_rate_parameters):
        barc, theta, sigma = ou_growth_rate_parameters
        bt, _, bs, ds, bgr, _ = self.get_value()
        if ds == None:
            raise ValueError('The division size has not been computed yet')
        else:
            tresh = np.log(ds/bs)
            time, dgr = sample_time_adaptive_method(tresh , eps, dtmin, bgr, barc, theta, sigma)
        return time, dgr

    def division(self):
        """ Divides a cell into two new daughters with the rule of equal mitosis """
        bt, dt, bs, ds, bgr, dgr = self.get_value()
        if ds == None:
            raise ValueError('The division size has not been computed yet')
        elif dt == None:
            raise ValueError('The life time has not been computed yet')
        else:
            c1 = Cell( dt, ds/2, dgr)
            c2 = Cell( dt, ds/2, dgr)
        return c1, c2

class GF_equal_mitosis():
    """ Object model of growth fragmentation with exponential growth
    Allows to simulate single samples of this model"""
    def __init__(self, ou_growth_rate_parameters ,division_rate_params):
        """
        Arguments:

        growth_rate : positive, float; growth_rate in the exponential growth
        division_rate : function, division_rate(x) is the probability that a polymer of size x divides.
        division_rate_params = (alpha, offset) : params of the parametrized K
        """

        self.ou_growth_rate_parameters = ou_growth_rate_parameters
        #self.K = f_division_rate
        self.div_params = division_rate_params

    def run(self, Tmax, init, itemax = 1e6, dtmin = 1e-4, eps = 0.05):
        alpha, offset = self.div_params
        ou_growth_rate_parameters = self.ou_growth_rate_parameters

        # Compute for the first cell
        init.division_size = init.sample_division_size_parametrized(alpha, offset)
        init.division_time, init.division_growth_rate = init.get_time(dtmin, eps,ou_growth_rate_parameters)   

        # Initialize loop parameters
        init_key = 0
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

                ofsp_key = counter
                all_cells[ofsp_key] = ofsp
                
                ofsp.division_size = ofsp.sample_division_size_parametrized(alpha, offset)

                elapsed_time, division_growth_rate = ofsp.get_time(dtmin, eps, ou_growth_rate_parameters)
                ofsp.division_time, ofsp.division_growth_rate = elapsed_time + ofsp.birth_time, division_growth_rate
                
                heapq.heappush(times_and_keys, (ofsp.division_time, ofsp_key))

        return all_cells, times_and_keys
            

        



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    
    growth_rate = 1
    theta = 1
    sigma = 1
    init_birth_size = 1

    ou_growth_rate_parameters = (growth_rate, theta, sigma)
    division_rate_params = offset, alpha = 0, 0
    Tmax = 3
    init = Cell(0,init_birth_size, growth_rate)

    Model = GF_equal_mitosis(ou_growth_rate_parameters = ou_growth_rate_parameters, division_rate_params = division_rate_params)

    dtmin = 1e-4
    eps = 0.05
    res, times = Model.run(Tmax, init, dtmin = dtmin, eps = eps)

    for cell_key in res:
        print(res[cell_key].birth_time)