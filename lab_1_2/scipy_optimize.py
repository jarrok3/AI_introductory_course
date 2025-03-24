#
# This file was implemented as reference for the manually-built solution
# ----------------------------------
# SCIPY RESULT CHECK
#

import numpy as np
from scipy.optimize import minimize
import problem_functions as fnc  

def find_min_1D(x0=0):
    """Finds the minimum of a one-dimensional function using scipy.optimize."""
    result = minimize(fnc.f, x0=x0, method="BFGS")
    return result.x[0], result.fun  

def find_min_2D(x0=(0, 0)):
    """Finds the minimum of a two-dimensional function using scipy.optimize."""
    result = minimize(lambda x: fnc.g(x[0], x[1]), x0=np.array(x0), method="BFGS")
    return result.x, result.fun 