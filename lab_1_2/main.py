import problem_functions as fnc
import paint_figure as paint
import sys

if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except IndexError as err:
        print(f"Expecting an argument in position 1, please submit your selection: \nselect 1 for 2D function, \nselect 2 for 3D function\nError message: {err}")
        sys.exit()
    
    if arg == "1":
        paint.visualize_function_2D(fnc.f,-5)
    elif arg == "2":
        paint.visualize_function_3D(fnc.g,3,3)
    else:
        print("Expecting a selection of 1 or 2")
        sys.exit()