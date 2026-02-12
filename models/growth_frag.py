

def sample_from_K(n_samples,K):
    """ Samples first hitting time of Poisson process with intesity K
    work in progress """
    return None

def sample_from_K_parametrized(n_samples, alpha = 0 , offset = 0):
    r""" Samples first hitting time with K(x) = 1_{x \ge offset} * (x - offset)^alpha"""

    U = np.random.rand(n_samples)
    return offset + ((1 + alpha) * -np.log(U) )**(1 / (1 + alpha))

class Equal_mitosis():
    """ Object model of growth fragmentation with exponential growth
    Allows to simulate single samples of this model"""
    def __init__(self,growth_rate,division_rate_params,f_division_rate = None):
        """
        Arguments:

        growth_rate : positive, float; growth_rate in the exponential growth
        division_rate : function, division_rate(x) is the probability that a polymer of size x divides.
        division_rate_params = (alpha, offset) : params of the parametrized K
        """

        self.c = growth_rate
        self.K = f_division_rate
        self.div_params = division_rate_params

    def run(self,Nmax,Tmax):
        time = 0 

        cell_sizes = np.full((Nmax, Tmax), np.nan)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt


    alpha = 10
    n_samples = 10000
    samples = sample_from_K_parametrized(n_samples, alpha)

    print(np.mean(samples))
    print(np.std(samples))