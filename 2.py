import numpy as np

def inverse_iteration_shift(A, lambda_0, shift, tol=1e-6, max_iter=1000):
    n = A.shape[0]
    x = np.random.rand(n)

    for _ in range(max_iter):
        # Вычисление матрицы (A - lambda*I)
        shifted_matrix = A - lambda_0 * np.eye(n)

        # Решение системы уравнений (A - lambda*I)*x = y
        y = np.linalg.solve(shifted_matrix, x)

        x_new = y / np.linalg.norm(y)

        # Вычисление нового приближенного собственного значения
        lambda_new = x_new.dot(A.dot(x_new))

        if np.abs(lambda_new - lambda_0) < tol:
            break

        lambda_0 = lambda_new + shift
        x = x_new

    return lambda_new, x_new


if __name__ == "__main__":
    A = np.loadtxt('input_matrix.txt')
    lambda_0, shift = np.loadtxt('input_parameters.txt')

    lambda_, x = inverse_iteration_shift(A, lambda_0, shift)

    # Вывод результатов в файл
    np.savetxt('output_result.txt', [lambda_, *x], header='Eigenvalue lambda and corresponding eigenvector x:')
