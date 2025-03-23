import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = (1 / m) * cost
    return total_cost



def gradient_descent(x, y, iterations, learning_rate, stopping_threshold=1e-6):
    current_weight = 0.1
    current_bias = 0.01

    n = len(x)

    costs = []
    weights = []
    previous_cost = None
    weight_derivative = 0
    bias_derivative = 0
    for i in range(iterations):
        y_predicted = (current_weight * x) + current_bias
        current_cost = compute_cost(x, y, current_weight, current_bias)
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break

        previous_cost = current_cost
        weights.append(current_weight)
        costs.append(current_cost)

        weight_derivative = (2 / n) * np.sum(x * (y_predicted - y))
        bias_derivative = (2 / n) * np.sum(y_predicted - y)

        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

        print(f"Iteration: {i}, Cost: {current_cost}, Weight: {current_weight}, Bias: {current_bias}")

    plt.figure(figsize=(8, 6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()

    return current_weight, current_bias

def main():
    x, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

    y = y.reshape(-1, 1)

    lr = 0.01

    estimated_weight, estimated_bias = gradient_descent(x, y, 2000, learning_rate=lr)

    print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

    Y_pred = estimated_weight * x + estimated_bias

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, marker='o', color='red')
    plt.plot([min(x), max(x)], [min(Y_pred), max(Y_pred)], color='blue', markerfacecolor='red',
             markersize=10, linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    main()