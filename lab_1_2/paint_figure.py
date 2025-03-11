import matplotlib.pyplot as plt
import numpy as np

def visualize_function_2D(f:callable, plot_step=1000, x_range=[-5,5]):
    """visualize function with one argument (2D)

    Args:
        f (callable): function of x
        plot_step (int, optional): Defaults to 1000.
        x_range (list, optional): Defaults to [-5,5].
    """
    # Vars + Values
    x = np.linspace(x_range[0],x_range[1],plot_step)
    Y = f(x)
    
    # Figure setup
    plt.figure(figsize=(8,6))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("One Arg Function Optimization")
    plt.plot(x,Y)
    plt.show()

def visualize_function_3D(g:callable, plot_step=1000, x_range=[-100,100]):
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
    
    # Figure setup
    plt.figure(figsize=(8,6))
    plt.pcolormesh(X1, X2, Z, cmap='viridis', shading='auto')
    plt.colorbar(label='Function Value (z)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Objective Function Visualization')
    plt.show()