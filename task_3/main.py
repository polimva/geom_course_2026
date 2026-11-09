import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


# Параметры задачи

M1 = 1.0
M2 = 1.0

L1 = 1.0
L2 = 1.0

G = 9.81

T0 = 0.0
T1 = 20.0

N_OUTPUT = 1000

RTOL = 1e-8
ATOL = 1e-8

Y0_A = np.array([
    np.pi / 2,
    0.0,
    np.pi / 2 + 0.01,
    0.0
])

DELTA_0 = 1e-6


# Система дифференциальных уравнений

def right_hand_side(t, y):
    theta1, omega1, theta2, omega2 = y

    delta = theta1 - theta2

    denominator = (
        2 * M1 + M2 - M2 * np.cos(2 * delta)
    )

    theta1_dot = omega1

    omega1_dot = (
        -G * (2 * M1 + M2) * np.sin(theta1)
        - M2 * G * np.sin(theta1 - 2 * theta2)
        - 2 * np.sin(delta) * M2
        * (
            omega2**2 * L2
            + omega1**2 * L1 * np.cos(delta)
        )
    ) / (L1 * denominator)

    theta2_dot = omega2

    omega2_dot = (
        2 * np.sin(delta)
        * (
            omega1**2 * L1 * (M1 + M2)
            + G * (M1 + M2) * np.cos(theta1)
            + omega2**2 * L2 * M2 * np.cos(delta)
        )
    ) / (L2 * denominator)

    return np.array([
        theta1_dot,
        omega1_dot,
        theta2_dot,
        omega2_dot
    ])


# Численное решение задачи Коши

def solve_motion(y0, rtol=RTOL, atol=ATOL):

    t_eval = np.linspace(T0, T1, N_OUTPUT)

    solution = solve_ivp(
        right_hand_side,
        (T0, T1),
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution


# Декартовы координаты грузов

def to_cartesian(solution):

    theta1 = solution.y[0]
    theta2 = solution.y[2]

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2


# Механическая энергия

def mechanical_energy(solution):

    theta1, omega1, theta2, omega2 = solution.y

    delta = theta1 - theta2

    kinetic = (
        0.5 * (M1 + M2) * L1**2 * omega1**2
        + 0.5 * M2 * L2**2 * omega2**2
        + M2 * L1 * L2 * omega1 * omega2 * np.cos(delta)
    )

    potential = (
        -(M1 + M2) * G * L1 * np.cos(theta1)
        - M2 * G * L2 * np.cos(theta2)
    )

    return kinetic + potential


# ЗАДАНИЕ 1. Траектория нижнего груза

def plot_lower_mass_trajectory(solution):

    _, _, x2, y2 = to_cartesian(solution)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(x2, y2)

    ax.set_title("Траектория нижнего груза")
    ax.set_xlabel("x₂, м")
    ax.set_ylabel("y₂, м")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    return fig


# ЗАДАНИЕ 1. Анимация двойного маятника

def create_animation(solution):

    x1, y1, x2, y2 = to_cartesian(solution)

    fig, ax = plt.subplots(figsize=(7, 7))

    limit = 1.1 * (L1 + L2)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.set_aspect("equal", adjustable="box")

    ax.set_title("Движение двойного маятника")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")

    ax.grid(True)

    rods, = ax.plot(
        [],
        [],
        "o-",
        linewidth=2,
        markersize=7
    )

    trail, = ax.plot([], [], "-", linewidth=1)

    time_label = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes
    )

    def init():

        rods.set_data([], [])
        trail.set_data([], [])
        time_label.set_text("")

        return rods, trail, time_label

    def update(frame):

        rods.set_data(
            [0, x1[frame], x2[frame]],
            [0, y1[frame], y2[frame]]
        )

        trail.set_data(
            x2[:frame + 1],
            y2[:frame + 1]
        )

        time_label.set_text(
            f"t = {solution.t[frame]:.2f} c"
        )

        return rods, trail, time_label

    animation = FuncAnimation(
        fig,
        update,
        frames=len(solution.t),
        init_func=init,
        interval=20,
        blit=True
    )

    return fig, animation


# ЗАДАНИЕ 2. Чувствительность к начальным условиям

def plot_sensitivity(solution_a, solution_b, y0_a, y0_b):

    _, _, x2_a, y2_a = to_cartesian(solution_a)
    _, _, x2_b, y2_b = to_cartesian(solution_b)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharex=True,
        sharey=True
    )

    axes[0].plot(x2_a, y2_a)

    axes[0].set_title(
        f"Случай A: θ₂(0) = {y0_a[2]:.8f} рад"
    )

    axes[0].set_xlabel("x₂, м")
    axes[0].set_ylabel("y₂, м")

    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True)

    axes[1].plot(x2_b, y2_b)

    axes[1].set_title(
        f"Случай B: θ₂(0) = {y0_b[2]:.8f} рад"
    )

    axes[1].set_xlabel("x₂, м")

    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True)

    fig.suptitle("Чувствительность к начальным условиям")

    fig.tight_layout(rect=(0, 0, 1, 0.93))

    return fig


# ЗАДАНИЕ 2. Относительное изменение энергии

def relative_energy_error(solution):

    energy = mechanical_energy(solution)

    initial_energy = abs(energy[0])

    if initial_energy == 0:
        raise ZeroDivisionError("Начальная энергия равна нулю")

    return np.abs(energy - energy[0]) / initial_energy


# ЗАДАНИЕ 2. Сравнение точности интегратора

def plot_energy_comparison(solution_1e8, solution_1e5):

    error_1e8 = relative_energy_error(solution_1e8)
    error_1e5 = relative_energy_error(solution_1e5)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogy(
        solution_1e8.t,
        error_1e8,
        label="rtol = atol = 10⁻⁸"
    )

    ax.semilogy(
        solution_1e5.t,
        error_1e5,
        label="rtol = atol = 10⁻⁵"
    )

    ax.set_title("Относительное изменение механической энергии")
    ax.set_xlabel("t, с")
    ax.set_ylabel("|E(t) − E(0)| / |E(0)|")

    ax.grid(True, which="both")
    ax.legend()

    print(
        "Максимальная ошибка энергии при 10⁻⁸:",
        f"{np.max(error_1e8):.6e}"
    )

    print(
        "Максимальная ошибка энергии при 10⁻⁵:",
        f"{np.max(error_1e5):.6e}"
    )

    return fig


# Главная функция

def main():

    # ЗАДАНИЕ 1

    solution_a = solve_motion(Y0_A)

    plot_lower_mass_trajectory(solution_a)

    animation = create_animation(solution_a)

    # ЗАДАНИЕ 2

    y0_b = Y0_A.copy()
    y0_b[2] += DELTA_0

    solution_b = solve_motion(
        y0_b,
        rtol=1e-8,
        atol=1e-8
    )

    plot_sensitivity(
        solution_a,
        solution_b,
        Y0_A,
        y0_b
    )

    # Проверка сохранения энергии

    solution_a_1e5 = solve_motion(
        Y0_A,
        rtol=1e-5,
        atol=1e-5
    )

    plot_energy_comparison(
        solution_a,
        solution_a_1e5
    )

    plt.show()


if __name__ == "__main__":
    main()