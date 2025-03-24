import numpy as np
import problem_functions as fnc

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
        

def gradient_solve(f:callable, step, threshold, *init_args, max_steps=1000, upper_threshold=1e5):
    X_INIT = np.array(init_args,dtype=float)
    print(f"Array init:\n{X_INIT}\n===================\n")
    
    path = [X_INIT.copy()]
    
    # Execute gradient descent
    for _ in range(max_steps):
        derivative = numerical_derivative(f,1e-5,X_INIT)
        
        print(f"Derivative: {derivative}")
        X_NEW = X_INIT - step * derivative
        
        path.append(X_NEW.copy())
        
        # Stop when minimum found
        if np.all(np.abs(X_NEW - X_INIT)<= threshold):
            break
        # Stop when minimum cannot be found
        if np.any(np.abs(X_NEW-X_INIT) > upper_threshold):
            break
        
        print(f"New x:\n{X_NEW}\n===========")
        X_INIT = X_NEW
        
    return X_INIT, np.array(path)
        
