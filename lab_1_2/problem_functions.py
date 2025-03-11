# Define problem functions
def f(x):
    """f(x) = 10x^4 + 3x^3 - 30x^2 + 10x"""
    return 10*x**4 + 3*x**3 - 30*x**2 + 10*x

def g(x1, x2):
    """g(x1, x2) = (x1 − 2)^4 + (x2 + 3)^4 + 2(x1 − 2)^2(x2 + 3)^2"""
    return (x1-2)**4 + (x2+3)**4 + 2*((x1-2)**2)*((x2+3)**2)
