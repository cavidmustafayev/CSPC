import numpy as np
from scipy.optimize import minimize, newton

def f(x):
    return (x - 3) ** 2 + 1

def df(x):
    return 2 * (x - 3)

def d2f(x):
    return 2

def gradient_descent_f(x0=0.0, lr=0.1, tol=1e-6, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        grad = df(x)
        if abs(grad) < tol:
            break
        x = x - lr * grad
    return x

x_gd = gradient_descent_f(x0=0.0)
print(f"Gradient Descent result: {x_gd:.2f}")

x_newton = newton(func=df, x0=0.0, fprime=d2f)
print(f"Newton's method result: {x_newton:.2f}")

res_slsqp = minimize(f, x0=[0.0], method="SLSQP")
print(f"SLSQP result: {res_slsqp.x[0]:.2f}\n")

def g(x):
    return x**4 - 3 * (x**2) + x + 5


def dg(x):
    return 4 * (x**3) - 6 * x + 1


def d2g(x):
    return 12 * (x**2) - 6


def gradient_descent_g(x0, lr=0.01, tol=1e-6, max_iter=10000):
    x = x0
    for _ in range(max_iter):
        grad = dg(x)
        if abs(grad) < tol:
            break
        x = x - lr * grad
    return x

def run_experiment(x0):
    x_gd = gradient_descent_g(x0)
    print(f"Gradient Descent result: x = {x_gd:.4f}, g(x) = {g(x_gd):.4f}")

    try:
        x_newton = newton(func=dg, x0=x0, fprime=d2g)
        second_deriv = d2g(x_newton)
        point_type = "Minimum" if second_deriv > 0 else "Maximum"
        print(
            f"Newton's method result: x = {x_newton:.4f}, g''(x) = {second_deriv:.4f} ({point_type})"
        )
    except Exception as e:
        print(f"Newton's method failed: {e}")

    res_slsqp = minimize(g, x0=[x0], method="SLSQP")
    print(
        f"SLSQP result:x = {res_slsqp.x[0]:.4f}, g(x) = {g(res_slsqp.x[0]):.4f}\n")

run_experiment(x0=0.0)
run_experiment(x0=2.0)