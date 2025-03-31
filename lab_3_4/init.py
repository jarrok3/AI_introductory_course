import numpy as np

def init_trivial(size=20):
    trivial = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            trivial[i, j] = (i + j) % 2
    return trivial
            

def initialize_population(pop_size, genome_length):
    """Generate a pop_size number of individuals with the accurate genome_length (vector)

    Args:
        pop_size (_type_): _description_
        genome_length (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.random.randint(2, size=(pop_size, genome_length))