
from collections.abc import Callable


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    eps: float,
) -> float:
    """Находит приближённый корень уравнения f(x) = 0 методом бисекции."""

    if eps <= 0:
        raise ValueError("Точность eps должна быть положительной.")

    if a >= b:
        raise ValueError("Левая граница a должна быть меньше правой b.")

    fa = f(a)
    fb = f(b)

    # Если корень находится на конце отрезка.
    if fa == 0:
        return a

    if fb == 0:
        return b

    # Проверяем наличие смены знака на концах отрезка.
    if fa * fb >= 0:
        raise ValueError(
            "На концах отрезка функция должна иметь разные знаки."
        )

    # Повторяем деление отрезка пополам.
    while b - a >= eps:
        c = (a + b) / 2
        fc = f(c)

        # Проверяем точное попадание в корень
        # или достижение требуемой точности по функции.
        if fc == 0 or abs(fc) < eps:
            return c

        # Оставляем половину отрезка со сменой знака.
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    # Длина отрезка стала меньше заданной точности.
    return (a + b) / 2


def main():
    # Пример: x^3 - x - 2 = 0.
    def f(x: float) -> float:
        return x**3 - x - 2

    a = 1.0
    b = 2.0
    eps = 1e-6

    root = bisection(f, a, b, eps)

    print("Метод бисекции")
    print("Уравнение: x^3 - x - 2 = 0")
    print(f"Начальный отрезок: [{a}, {b}]")
    print(f"Точность: {eps}")
    print(f"Найденный корень: {root:.10f}")
    print(f"Проверка f(x): {f(root):.10g}")


if __name__ == "__main__":
    main()
