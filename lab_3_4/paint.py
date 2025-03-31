import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def paint_solution(individual,max_score, title = "Best individual score: "):
    # Generate visualization
    adjacent_mask = np.zeros_like(individual, dtype=bool)

    # Check if 0 is a point (similiar to evaluate)
    adjacent_mask[:-1, :] |= (individual[1:, :] == 1)   
    adjacent_mask[1:, :] |= (individual[:-1, :] == 1)
    adjacent_mask[:, :-1] |= (individual[:, 1:] == 1)  
    adjacent_mask[:, 1:] |= (individual[:, :-1] == 1)
    
    color_map = np.where(individual == 1, 1, 0)  
    color_map[(individual == 0) & adjacent_mask] = 2  
    
    # Define custom colormap
    cmap = mcolors.ListedColormap(["yellow", "green", "lightgreen"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.xticks(np.arange(20))
    plt.yticks(np.arange(20))
    for i in np.arange(0.5, 20, 1):
        plt.axhline(i, color='black', linestyle='-', linewidth=0.5)
        plt.axvline(i, color='black', linestyle='-', linewidth=0.5)
    plt.imshow(color_map, cmap=cmap, norm=norm)
    plt.title(f"{title} {max_score}")
    plt.show()