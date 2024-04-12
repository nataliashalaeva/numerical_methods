import numpy as np
import matplotlib.pyplot as plt

def ode_system(t, y):
    dx = 2 * y[0] - 5 * y[1] + 3
    dy = 5 * y[0] - 6 * y[1] + 1
    return np.array([dx, dy])

def adams_method(f, y0, t, h):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(4):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h/2, y[i] + k1/2)
        k3 = h * f(t[i] + h/2, y[i] + k2/2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    for i in range(4, n-1):
        y_pred = y[i] + h/720 * (251*f(t[i+1], y[i+1]) + 646*f(t[i], y[i]) - 264*f(t[i-1], y[i-1]) + 106*f(t[i-2], y[i-2]) - 19*f(t[i-3], y[i-3]))
        y[i+1] = y[i] + h/720 * (251*f(t[i+1], y_pred) + 646*f(t[i], y[i]) - 264*f(t[i-1], y[i-1]) + 106*f(t[i-2], y[i-2]) - 19*f(t[i-3], y[i-3]))

    return y

# Настройка параметров
y0 = np.array([6, 5])
num_steps = 100
t_start, t_end = 0, 5
h = 0.2
t_values = np.arange(t_start, t_end + h, h)

# Получение численных решений с шагами h и h/2
y_h = adams_method(ode_system, np.array([6, 5]), t_values, h)
y_h_half = adams_method(ode_system, np.array([6, 5]), t_values, h/2)

# Аналитическое решение
def exact_solution(t):
    dx = 5 * np.exp(-2*t) * np.cos(3*t) + 1
    dy = np.exp(-2*t) * (4 * np.cos(3*t) + 3 * np.sin(3*t)) + 1
    return np.array([dx, dy])

exact_values = np.array([exact_solution(t) for t in t_values])

# Запись результатов в файл
with open("results.txt", "w",encoding="utf-8") as file:
    file.write("t\tШаг h - x\tШаг h - y\tШаг h/2 - x\tШаг h/2 - y\tАналитическое решение - x\tАналитическое решение - y\n")
    for i in range(len(t_values)):
        line = f"{t_values[i]}\t{y_h[i, 0]}\t{y_h[i, 1]}\t{y_h_half[i, 0]}\t{y_h_half[i, 1]}\t{exact_values[i, 0]}\t{exact_values[i, 1]}\n"
        file.write(line)

# Графическое представление результатов
plt.plot(t_values, y_h[:, 0], label='Шаг h - x')
plt.plot(t_values, y_h[:, 1], label='Шаг h - y')

plt.plot(t_values, y_h_half[:, 0], label='Шаг h/2 - x', linestyle='--')
plt.plot(t_values, y_h_half[:, 1], label='Шаг h/2 - y', linestyle='--')

plt.plot(t_values, exact_values[:, 0], label='Аналитическое решение - x', linestyle=':')
plt.plot(t_values, exact_values[:, 1], label='Аналитическое решение - y', linestyle=':')

plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.show()
