import problem_functions as fnc
import paint_figure as paint
import scipy_optimize as scp
import sys

if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except IndexError as err:
        print(f"Expecting an argument in position 1, please submit your selection: \nselect 1 for 2D function, \nselect 2 for 3D function\nError message: {err}")
        sys.exit()
    
    if arg == "1":
        try:
            x_init = float(sys.argv[2])
        except IndexError as err:
            print(f"Expecting a starting point at position 2, Error message: {err}")
            sys.exit()
            
        paint.visualize_function_2D(fnc.f,x_init)
        
        # Check through other libraries if the result was correct
        min_x, min_f = scp.find_min_1D(x0=x_init)
        print(f"[SciPy] 1D Function Minimum: x = {min_x:.6f}, f(x) = {min_f:.6f}\n==========")
        
    elif arg == "2":
        try:
            x_1 = float(sys.argv[2])
            x_2 = float(sys.argv[3])
        except IndexError as err:
            print(f"Expecting a starting vector at position 2 and 3, Error message: {err}")
            sys.exit()
        
        paint.visualize_function_3D(fnc.g,x_1,x_2)
        
        # Check through other libraries if the result was correct
        min_x2D, min_g = scp.find_min_2D(x0=(x_1, x_2))
        print(f"[SciPy] 2D Function Minimum: x1 = {min_x2D[0]:.6f}, x2 = {min_x2D[1]:.6f}, g(x1, x2) = {min_g:.6f}")

    else:
        print("Expecting a selection of 1 or 2")
        sys.exit()