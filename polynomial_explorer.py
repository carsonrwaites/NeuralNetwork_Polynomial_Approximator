import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

X = np.linspace(-4, 4, 1000).astype(np.float32)
x = sp.Symbol('x')

## WRITE POLYNOMIALS HERE
polys = [2*x**2 + 3*x - 5,
         x**3 - 6*x**2 + 4*x + 8,
         x**4 - 8*x**2 + 3*x + 6,
         x**5 - 10*x**3 + 6*x**2 + 8*x - 4,
         x**6 - 12*x**4 + 3*x**3 + 20*x**2 - 5*x - 8,
         x**7 - 14*x**5 + 4*x**3 + 10*x**2 - 7*x + 3,
         x**8 - 16*x**6 + 5*x**4 - 12*x**2 + 4*x + 6,
         x**9 - 18*x**7 + 7*x**5 - 14*x**3 + 9*x - 2]

for poly in polys:
    expr = poly

    latex_str = sp.latex(expr)
    f = sp.lambdify(x, expr, "numpy")
    Y = f(X)

    plt.plot(X, Y, label=f"$f(x)={latex_str}$")
    plt.legend()
    plt.show()