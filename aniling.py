import numpy as np
import random

def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2*np.pi*xi)) for xi in x])

def sim_ann(d=2, T=1000, alpha=0.99, max_iter=10000):
    cs = np.random.uniform(-5.12, 5.12, d)  # current solution
    ce = rastrigin(cs)                     # current energy
    bs = cs.copy()                         # best solution
    be = ce                                # best energy

    for i in range(max_iter):
        ns = cs + np.random.uniform(-0.5, 0.5, d)  # new solution
        ns = np.clip(ns, -5.12, 5.12)
        ne = rastrigin(ns)                         # new energy
        
        diff = ne - ce
        if diff < 0 or np.exp(-diff / T) > random.random():
            cs, ce = ns, ne
        
        if ce < be:
            bs, be = cs.copy(), ce
        
        T *= alpha
        if T < 1e-8:
            break

    return bs, be

if __name__ == "_main_":
    sol, val = sim_ann()
    print("Best Solution:", sol)
    print("Best Value:", val)