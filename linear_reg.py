import csv
import matplotlib.pyplot as plt


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


class LinearRegression:
    def __init__(self):
        self._b0, self._b1 = None, None

    def fit(self, x_train, y_train):
        x_mean, y_mean = mean(x_train), mean(y_train)
        self._b1 = covariance(x_train, x_mean, y, y_mean) / variance(x_train, x_mean)
        self._b0 = y_mean - self._b1 * x_mean

    def predict(self, x_pred):
        if self._b0 is not None and self._b1 is not None:
            y_pred = [self._b0 + self._b1*x_pred[i] for i in range(len(x_pred))]
            return y_pred
        else:
            raise Exception('Coefficients of linear regression are not determinated')


if __name__ == "__main__":
    csv_path = "data.csv"
    # x1, x2, y = [], [], []
    with open(csv_path, "r") as f_obj:
        x1, x2, y = csv_reader(f_obj)

    lr1 = LinearRegression()
    lr1.fit(x1, y)
    xp1 = [min(x1), max(x1)]
    yp1 = lr1.predict(xp1)

    lr2 = LinearRegression()
    lr2.fit(x2, y)
    xp2 = [min(x2), max(x2)]
    yp2 = lr2.predict(xp2)

    plt.scatter(x1, y, edgecolors='red')
    plt.scatter(x2, y)

    plt.plot(xp1, yp1)
    plt.plot(xp2, yp2)

    plt.show()
