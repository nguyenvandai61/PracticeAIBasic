
# generate data
# list of points
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.1, .0], [.0, .1]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)

plt.grid()

plt.plot(X0[0], X0[1], 'ro')     # data
plt.plot(X1[0], X1[1], 'y^')
plt.axis([0, 6, 0, 6])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')




# Xbar
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)


def h(w, x):
    return np.sign(np.dot(w.T, x))


def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)


def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    while True:
        # mix data
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi:  # misclassified point
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi
                w.append(w_new)

        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)


d = X.shape[0]
w_init = np.random.randn(d, 1)

(w, m) = perceptron(X, y, w_init)
w_res = w[len(w)-1]
x0 = np.linspace(0, 6, 2)
y0 = (w_res[0] + w_res[1]*x0)/(-w_res[2])
plt.plot(x0, y0)
#print(np.array(w), "\n", m, "\n")
plt.show()