import numpy as np
from scipy.integrate import simpson, trapezoid

def rec_curve(predict, exact, p=2, return_aoc=False):
    epsilon = np.power(np.abs(predict - exact), p)

    error, accuracy = _rec_epsilon(epsilon)

    if return_aoc:
        aoc = np.max(error) - trapezoid(accuracy, error)

        return(error, accuracy, aoc)
    else:
        return(error, accuracy)

def _rec_epsilon(epsilon):
    epsilon = epsilon
    epsilon = np.sort(epsilon)
    # weights =
    eps_prev = 0
    correct = 0
    m = len(epsilon)


    error = []
    accuracy = []

    for i, eps in enumerate(epsilon):
        if eps > eps_prev:
            accuracy.append(correct/m)
            error.append(eps_prev)
            eps_prev = eps
        correct += 1
    return(error, accuracy)
