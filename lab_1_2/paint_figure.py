import matplotlib.pyplot as plt
import numpy as np
import gradient_solve as grad

def visualize_function_2D(f:callable, init_point, plot_step=1000, x_range=[-5,5]):
    """visualize function with one argument (2D)

    Args:
        f (callable): function of x
        init_point (float): initial starting point for the gradient descent
        plot_step (int, optional): Defaults to 1000.
        x_range (list, optional): Defaults to [-5,5].
    """
    # Vars + Values
    x = np.linspace(x_range[0],x_range[1],plot_step)
    Y = f(x)
    minimum, path = grad.gradient_solve(f, 1e-4, 1e-7, init_point, max_steps=1000)
    min_value = f(minimum)
    
    # Figure setup
    plt.figure(figsize=(8,6))
    
    # Plot function and gradient descent
    plt.plot(x, Y, label="f(x)", color="blue") 
    plt.plot(path, f(path), label="Descent Path", color="red")
    
    # Plot minimum point
    plt.scatter(minimum, min_value, color="black", marker="o", label="Minimum Point", zorder=3)
    plt.text(minimum, min_value, f"({minimum}, {min_value})", fontsize=10, verticalalignment="top", horizontalalignment="left", color="black")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("One Arg Function Optimization")
    plt.legend()
    plt.show()

def visualize_function_3D(g:callable, init_x1, init_x2, plot_step=1000, x_range=[-10,10]):
    """visualize function with two arguments (3D)

    Args:
        g (callable): function of x1, x2
        plot_step (int, optional): Defaults to 1000.
        x_range (list, optional): Defaults to [-100,100].
    """
    # Vars + Values
    x1 = np.linspace(x_range[0], x_range[1], plot_step)
    x2 = np.linspace(x_range[0], x_range[1], plot_step)
    X1, X2 = np.meshgrid(x1,x2)
    Z = g(X1,X2)
    minimum, path = grad.gradient_solve(g, 1e-3, 1e-5, init_x1, init_x2, max_steps=100000)
    
    path_x1 = path[:, 0]
    path_x2 = path[:, 1]
    
    # Figure setup
    plt.figure(figsize=(8,6))
    
    # Plot function and gradient descent
    plt.pcolormesh(X1, X2, Z, cmap='viridis', shading='auto')
    plt.colorbar(label='Function Value (z)')
    plt.plot(path_x1, path_x2, color="red", label="Descent Path")
    
    # Plot minimum point
    plt.scatter(minimum[0], minimum[1], color="black", marker="o", s=50, label="Minimum Point", zorder=3)
    plt.text(minimum[0], minimum[1], f"({minimum[0]:.4f}, {minimum[1]:.4f})", 
             fontsize=10, verticalalignment="top", horizontalalignment="left", color="black")

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Objective Function Visualization')
    plt.show()