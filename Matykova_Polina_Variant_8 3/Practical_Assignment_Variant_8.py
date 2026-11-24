import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def find_points():
    x, y = sp.symbols("x y")
    alpha, beta, delta, gamma = sp.symbols("alpha beta delta gamma")

    eq1 = alpha * x - beta * x * y
    eq2 = delta * x * y - gamma * y
    points = sp.solve((eq1, eq2), (x, y), dict=True)

    print("Stationary points:")
    for point in points:
        print(f"({point[x]}, {point[y]})")

    values = {alpha: 1.0, beta: 0.1, delta: 0.075, gamma: 1.5}
    point = next(point for point in points if point[x] != 0 and point[y] != 0)
    point = (float(point[x].subs(values)), float(point[y].subs(values)))
    print(f"Nontrivial point for given values: {point}")
    return point


def system(t, z):
    x, y = z
    dx = 1.0 * x - 0.1 * x * y
    dy = 0.075 * x * y - 1.5 * y
    return [dx, dy]


def solve_model():
    time = np.linspace(0, 30, 500)
    sol = solve_ivp(system, (0, 30), (10, 5), t_eval=time)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def draw_graphs(sol, point):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(sol.t, sol.y[0], label="x(t)")
    ax[0].plot(sol.t, sol.y[1], label="y(t)")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Population")
    ax[0].set_title("Populations over time")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(sol.y[0], sol.y[1], label="y(x)")
    ax[1].scatter(point[0], point[1], color="red", label="Stationary point")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Phase trajectory")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    plt.show()


def main():
    point = find_points()
    sol = solve_model()
    draw_graphs(sol, point)


if __name__ == "__main__":
    main()
