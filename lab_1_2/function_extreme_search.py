import os
import numpy as np
import matplotlib.pyplot as plt

# Use either a direct gradient or a numerical differentiation one (approximation through f'(x)=[f(x+e)-f(x-e)] / 2)

class Problem():
    def __init__(self):
        pass
        
        
class Solver():
    def __init__(self):
        pass
    
# Define problem functions
def f(x):
    """f(x) = 10x^4 + 3x^3 - 30x^2 + 10x"""
    return 10*x**4 + 3*x**3 - 30*x**2 + 10*x

def g(x1, x2):
    """g(x1, x2) = (x1 − 2)^4 + (x2 + 3)^4 + 2(x1 − 2)^2(x2 + 3)^2"""
    return (x1-2)**4 + (x2+3)**4 + 2*((x1-2)**2)*((x2+3)**2)

if __name__ == "__main__":
    # Figure setup
    fig, (ax1, ax2) = plt.subplots(2,1)
    x0=np.linspace(-5,5,100)
    
    # Subplot f(x)
    ax1.set_ylabel("f(x)")
    ax1.set_xlabel("x")
    ax1.plot(x0,f(x0))
    
    # Subplot g(x)
    ax2.set_ylabel("g(x)")
    ax2.set_xlabel("x")
    
    # Show
    plt.show()