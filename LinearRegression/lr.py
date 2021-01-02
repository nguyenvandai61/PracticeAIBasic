import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = []
        self.Xbar = []

    def linear_regression(self):
        one = np.ones((X.shape[0], 1))
        self.Xbar = np.concatenate((one, X), axis = 1)
        self.w = self.calc_weights()


    def calc_weights(self):
        A = np.dot(self.Xbar.T, self.Xbar)
        b = np.dot(self.Xbar.T, self.y)
        w = np.dot(np.linalg.pinv(A), b)
        print('w = ', w)
        return w
    def drawFigure(self):
        # Preparing the fitting line
        w_0 = self.w[0][0]
        w_1 = self.w[1][0]
        x0 = np.linspace(145, 185, 2)
        y0 = w_0 + w_1*x0

        plt.plot(X.T, self.y.T, 'ro')     # data
        plt.plot(x0, y0)               # the fitting line
    


if __name__ == "__main__":
    # height (cm)
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    # weight (kg)
    y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    # Visualize data
    LR = LinearRegression(X, y)
    LR.linear_regression()
    LR.drawFigure()
    # Drawing the fitting line
    plt.axis([140, 190, 45, 75])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
        
    plt.show()