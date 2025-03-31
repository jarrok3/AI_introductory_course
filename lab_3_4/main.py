import sys
from genetic_algorithm import genetic_algorithm
from paint import paint_solution
from init import init_trivial

if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except IndexError as err:
        print(f"Expecting an argument in position 1, please submit your selection: \n-t for tests, \n-m for single case\nError message: {err}")
        sys.exit()
    
    if arg == "-m":
        t = init_trivial()              
        best_individual, best_individual_fitness = genetic_algorithm()
        b = best_individual.reshape(20, 20)
        
        print("Best solution found:")
        print(b)
        print(f"Best solution score: {best_individual_fitness}")
        
        paint_solution(b,best_individual_fitness)
        paint_solution(t, 200, title="Trivial solution score: ")
        
    if arg == "-t":
        res = []
        for i in range(150):
            bes, bes_in = genetic_algorithm()
            res.append(bes_in)
        
        with open("test.txt", "w") as file:
            for item in res:
                file.write(f"{item}\n")
    
