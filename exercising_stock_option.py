import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt

# Variables
S0 = 10 # Market price
r = .04 # Interest free rate.
sigma = .2 # Volatility
T = .5 
N = 20
delta_t = T/N

def std_normal(n):
    """
    Returns a numpy array containing n values from a normal distribution.
    """
    result = []
    for i in range(0, int(n/2)):
        two = two_normals()
        result.append(two[0])
        result.append(two[1])
    return result

def two_normals():
    """
    Polar method for generating a standard normal Random Variable
    -------------------------------------------------------------
    Returns a pair of numbers that are distributed according to a
    Standard Normal Distribution.
    """
    while(True):
        # Generate V1 and V2 until we obtain one 
        # that is contained in the circle  of radius 1
        unif1 = np.random.uniform(0,1)
        unif2 = np.random.uniform(0,1)
        V1 = 2*unif1 - 1 # uniform on (-1,1)
        V2 = 2*unif2 - 1 # uniform on (-1,1)
        S = V1**2 + V2**2
        if (S<1): break
    X = np.sqrt( (-2*np.log(S))/S ) * V1
    Y = np.sqrt( (-2*np.log(S))/S ) * V2
    return (X,Y)

rnd_normals = std_normal(N)

# Simulation Step
Sh = [] # Simulated Price
S = [] # Real Price
for rand in rnd_normals:
    
    # Discrete version of the SDE:
    Ph = S0*( 1 + r*delta_t + sigma*rand*np.sqrt(delta_t) ) 
    # Formula for the exact solution of the SDE:
    P = S0*np.exp( (r-.5*sigma**2)*delta_t + sigma*rand*np.sqrt(delta_t) )
    
    Sh.append(Ph)
    S.append(P)

# Plotting Step
time = np.arange(1,N+1)
fig = pt.figure(figsize=(11,5))
plt = fig.add_subplot(111)
plt.plot(time, Sh, 'r-', label = 'Numerical')
plt.plot(time, S, 'b-', label = 'Exact')
plt.legend(loc='lower left')
pt.show()