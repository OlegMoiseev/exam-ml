import csv
import matplotlib.pyplot as plt
import numpy as np


def csv_reader(file_obj):
    reader = csv.reader(file_obj)
    x1, x2, y = [], [], []
    for row in reader:
        x1.append(float(row[1]))
        x2.append(float(row[2]))
        y.append(float(row[4]))
    return x1, x2, y


# Calculate the mean value of a list of numbers
def mean(values):
    return sum(values) / float(len(values))


# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


# Calculate the variance of a list of numbers
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])


def rmse(y, y_pred):
    return (np.sum((y_pred - y) ** 2) / float(len(y)))**0.5


class LinearRegression:
    def __init__(self):
        self._b0, self._b1 = None, None

    def fit(self, x_train, y_train):
        x_mean, y_mean = mean(x_train), mean(y_train)
        self._b1 = covariance(x_train, x_mean, y_train, y_mean) / variance(x_train, x_mean)
        self._b0 = y_mean - self._b1 * x_mean

    def get_params(self):
        return self._b0, self._b1

    def predict(self, x_pred):
        if self._b0 is not None and self._b1 is not None:
            y_pred = [self._b0 + self._b1*x_pred[i] for i in range(len(x_pred))]
            return y_pred
        else:
            raise Exception('Coefficients of linear regression are not determinated')


if __name__ == "__main__":
    # Create elements for learning
    X = 4 * np.random.rand(1000, 1)
    Y = 2 + 3 * X + np.random.randn(1000, 1)

    # Create elements for testing
    X_test = 4 + 4 * np.random.rand(1000, 1)
    Y_test = 2 + 3 * X_test + np.random.randn(1000, 1)

    # Fit regression
    lr1 = LinearRegression()
    lr1.fit(X, Y)

    # Draw main trend
    xp1 = [min(X), max(X)]
    yp1 = lr1.predict(xp1)
    plt.scatter(X, Y, c='red')
    plt.scatter(X_test, Y_test)

    # Use regression
    Y_pred = lr1.predict(X_test)

    # Calculate RMSE
    print('rmse: ', rmse(Y_test, Y_pred))
    plt.plot(xp1, yp1)

    plt.show()
