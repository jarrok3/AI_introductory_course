import numpy as np
import random
from init import initialize_population
from evaluate import evaluate

def roulette_selection(population, fitness_values):
    """Select members of the population based on their probabilities. Higher score equals higher probability (weight system).

    Args:
        population (_type_): _description_
        fitness_values (_type_): _description_

    Returns:
        _type_: _description_
    """
    total_fitness = np.sum(fitness_values)
    q = fitness_values/total_fitness
    q_max = np.max(q)
    q_min = np.min(q)
    probabilities = (q - q_min) / (q_max - q_min)
    sum_prob = sum(probabilities)
    weight_prob = probabilities/sum_prob
    # Select best individuals to create a new population
    selected = np.random.choice(len(population), size=len(population), p=weight_prob)
    
    return population[selected]

def single_point_crossover(parent1, parent2, crossover_p, mutation_rate):
    """creates two children with genomes from both parents

    Args:
        parent1 (_type_): _description_
        parent2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Choose crossover point randomly
    if random.random() <= crossover_p:
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
    else:
        child1 = parent1
        child2 = parent2
        
    child1 = mutate(child1, mutation_rate)
    child2 = mutate(child2, mutation_rate)
    return child1, child2

def mutate(individual, mutation_rate):
    """Randomly mutate signle genes within genomes of an individual

    Args:
        individual (_type_): _description_
        mutation_rate (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Mutate random genes in an individual's genome
    for i in range(len(individual)):
        if random.random() <= mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm(pop_size=100, genome_length=400, generations=100, mutation_rate=0.05, crossover_p=0.8):
    # init population and best individual
    population = initialize_population(pop_size, genome_length)
    fitness_values = evaluate(population)
    
    max_fitness = np.max(fitness_values)
    max_fitness_individual = population[np.argmax(max_fitness)]
    fitness_per_iter = []
    
    # do until you reach max iterations
    for generation in range(generations):
        selected_population = roulette_selection(population, fitness_values)
        
        new_population = []
        # pair selected population members and their DNA into new individuals
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1, child2 = single_point_crossover(parent1, parent2, crossover_p,mutation_rate)
            # mutation in new generations
            new_population.append(child1)
            new_population.append(child2)
        
        # new gen
        population = np.array(new_population)
        fitness_values = evaluate(population)
        best_fitness = np.max(fitness_values)
        fitness_per_iter.append(best_fitness)
        #print(f"Generation {generation+1}: Best Fitness = {best_fitness}")
        if best_fitness > max_fitness:
            max_fitness = best_fitness
            max_fitness_individual = population[np.argmax(max_fitness)]
    
    return max_fitness_individual, max_fitness, fitness_per_iter
