import os
import numpy as np
import matplotlib.pyplot as plt

class Problem():
    def __init__(self):
        pass
        
        
class Solver():
    def __init__(self):
        pass
    

def f(x):
    """f(x) = 10x^4 + 3x^3 - 30x^2 + 10x"""
    return (10*x^4 + 3*x^3 - 30*x^2 + 10*x)

def g(x1,x2):
    """g(x) =  (x1 − 2)^4 + (x2 + 3)^4 + 2(x1 − 2)^2(x2 + 3)^2"""
    return (x1-2)^4 + (x2+3)^4 + 2*((x1-2)^2)*((x2+3)^2)

if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(2,1)
    x0=np.arange(-5,5,1)
    print (x0)
    
    ax1.set_ylabel("f(x)")
    ax1.set_xlabel("x")
    ax1.plot(x0,f(x0))
    
    ax2.set_ylabel("g(x)")
    ax2.set_xlabel("x")
    plt.show()