import numpy as np
import problem_functions as fnc
import time

# Define numerical derivative for a coordinate array
def numerical_derivative(f:callable, h, coordinates):
    # Init derivative with number of dimensions = number of arguments
    derivative = np.zeros_like(coordinates,dtype=float)
    
    # Central approximation
    for i in range(len(coordinates)):  
        x_forward = coordinates.copy()
        x_backward = coordinates.copy()

        x_forward[i] += h # f(x+h,*x)
        x_backward[i] -= h # f(x-h,*x)

        derivative[i] = (f(*x_forward) - f(*x_backward)) / (2 * h)
        
    return derivative
        

def gradient_solve(f:callable, step, threshold, *init_args, max_steps=10000, upper_threshold=1e5):
    """_summary_

    Args:
        f (callable): _description_
        step (_type_): _description_
        threshold (_type_): _description_
        max_steps (int, optional): _description_. Defaults to 1000.
        upper_threshold (_type_, optional): _description_. Defaults to 1e5.

    Returns:
        _type_: _description_
    """
    
    X_INIT = np.array(init_args,dtype=float)

    path = [X_INIT.copy()]
    
    start_time = time.time()
    
    # Execute gradient descent
    for iteration_step in range(max_steps):
        derivative = numerical_derivative(f,1e-5,X_INIT)
        
        # print(f"Derivative: {derivative}")
        X_NEW = X_INIT - step * derivative
        
        path.append(X_NEW.copy())
        
        # Stop when minimum found
        if np.all(np.abs(X_NEW - X_INIT)<= threshold):
            print(f"Found minimum in {time.time()-start_time:.4f}s")
            print(f"Minimum x:{X_INIT}")
            break
        # Stop when minimum cannot be found
        if np.any(np.abs(X_NEW-X_INIT) > upper_threshold):
            print("No min found")
            break
        
        # print(f"New x:\n{X_NEW}\n===========")
        X_INIT = X_NEW
        
    print(f"Total number of iterations: {iteration_step}")
    return X_INIT, np.array(path)
        
