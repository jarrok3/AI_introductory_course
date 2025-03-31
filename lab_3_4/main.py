import numpy as np
import random
from evaluate import evaluate

def initialize_population(pop_size, genome_length):
    """Generate a pop_size number of individuals with the accurate genome_length (vector)

    Args:
        pop_size (_type_): _description_
        genome_length (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.random.randint(2, size=(pop_size, genome_length))

def roulette_selection(population, fitness_values):
    """Select members of the population based on their probabilities. Higher score equals higher probability (weight system).

    Args:
        population (_type_): _description_
        fitness_values (_type_): _description_

    Returns:
        _type_: _description_
    """
    total_fitness = np.sum(fitness_values)
    probabilities = fitness_values / total_fitness
    # Select best individuals to create a new population
    selected = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[selected]

def single_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm(pop_size=100, genome_length=400, generations=100, mutation_rate=0.01):
    population = initialize_population(pop_size, genome_length)
    
    # do until you reach max iterations
    for generation in range(generations):
        fitness_values = evaluate(population)
        selected_population = roulette_selection(population, fitness_values)
        
        new_population = []
        # pair selected population members and their DNA into new individuals
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1, child2 = single_point_crossover(parent1, parent2)
            # mutation in new generations
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = np.array(new_population)
        best_fitness = np.max(fitness_values)
        print(f"Generation {generation+1}: Best Fitness = {best_fitness}")
    
    return population[np.argmax(fitness_values)]

if __name__ == "__main__":
    best_solution = genetic_algorithm()
    print("Best solution found:")
    print(best_solution.reshape(20, 20))
