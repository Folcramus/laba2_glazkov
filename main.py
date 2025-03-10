import numpy as np
# Определяем функцию Phi(x, y)
def phi(x, y):
    return (np.cos(x - 1) + y - 0.5) ** 2 + (x - np.cos(y) - 3) ** 2

def grad_phi(x, y):
    df_dx = -2 * (np.cos(x - 1) + y - 0.5) * np.sin(x - 1) + 2 * (x - np.cos(y))
    df_dy = 2 * (np.cos(x - 1) + y - 0.5) - 2 * (x - np.cos(y)) * np.sin(y)
    return np.array([df_dx, df_dy])
# Метод скорейшего спуска
def steepest_descent(x0, y0, tol=1e-6, max_iter=1000):
    x, y = x0, y0 #начальное приближение
    for i in range(max_iter):
        x_old, y_old = x, y
        grad = grad_phi(x, y)

        # Вычисляем lambda_k
        lambda_k = phi(x, y) / np.dot(grad, grad)

        # Делаем шаг
        x = x - lambda_k * grad[0]
        y = y - lambda_k * grad[1]

        # Проверяем условие останова по относительной разнице (вычисляем градиент)
        delta_num = abs(x - x_old) + abs(y - y_old)  # это L1-норма разности
        delta_den = abs(x_old) + abs(y_old)  # это L1-норма старой точки
        if delta_den == 0:  # чтобы избежать деления на 0
            delta = delta_num
        else:
            delta = delta_num / delta_den

        if delta < 0.00001:
            break

    return x, y


# Начальные приближения
x0, y0 = 1.0, 1.0
# Решение системы уравнений
x_sol, y_sol = steepest_descent(x0, y0)
print("Начальные приближения:",x0,y0)
print(f"Решение: x = {x_sol}, y = {y_sol}\n")

# Начальные приближения
x0, y0 = -1.0, -1.0
# Решение системы уравнений
x_sol, y_sol = steepest_descent(x0, y0)
print("Начальные приближения:",x0,y0)
print(f"Решение: x = {x_sol}, y = {y_sol}")