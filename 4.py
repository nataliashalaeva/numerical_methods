import numpy as np
import matplotlib.pyplot as plt


def p(x):
    return 0


def q(x):
    return 1


def f(x):
    return 1


def gauss(A, B, X):
    n = len(B)  # Используем длину вектора B

    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            B[i] -= factor * B[k]

    X[n - 1] = B[n - 1] / A[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        sum_ax = np.dot(A[i][i + 1:n], X[i + 1:n])
        X[i] = (B[i] - sum_ax) / A[i][i]


alpha0 = 2
alpha1 = 2.5
Ac = 0
beta0 = 3
beta1 = 3.4
Bc = 5
a0 = 0.1
b0 = 1.3
n = 10
#шаг
h = (b0 - a0) / n
#cоздаем массив X, который содержит координаты узлов сетки от a0 до b0 с шагом h.
X = np.linspace(a0, b0, n + 1)

A = np.zeros((n + 1, n + 1))
B = np.zeros(n + 1)

# Заполнение матрицы и вектора для внутренних узлов
for i in range(1, n):
    A[i][i - 1] = 1 / h ** 2 - p(X[i]) / (2 * h)
    A[i][i] = -2 / h ** 2 + q(X[i])
    A[i][i + 1] = 1 / h ** 2 + p(X[i]) / (2 * h)
    B[i] = f(X[i])

# Граничные условия на левом конце (точность O(h))
A[0][0] = alpha0 - alpha1 / h
A[0][1] = alpha1 / h
B[0] = Ac

# Граничные условия на правом конце (точность O(h))
A[n][n - 1] = -beta1 / h
A[n][n] = beta0 + beta1 / h
B[n] = Bc

X1 = np.zeros(n + 1)
gauss(A, B, X1)

# Аналитическое решение
analytical_solution = -np.sin(X)-2* np.cos(X)+3

print("Матрица A:")
print(A)

print("\nВектор B:")
print(B)

print("\nРешение X1:")
print(X1)

plt.plot(X, X1+1, marker='o', linestyle='-', color='b', label='Численное решение')
plt.plot(X, analytical_solution, marker='x', linestyle='--', color='r', label='Аналитическое')
plt.xlabel('x')
plt.title('Решение')
plt.legend()
plt.grid(True)
plt.show()
