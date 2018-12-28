import numpy as np

def predict(X, w):
    return np.matmul(X, w)

def loss (X, Y, w):
    predictions = predict(X, w)
    return np.average((predictions - Y) ** 2)

def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

def train(X, Y, iterations, learning_rate):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iterations: %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * learning_rate
    return w

if __name__ == "__main__":
    X1, X2, X3, Y = np.loadtxt("examples.txt", skiprows=1, unpack=True)
    X = np.column_stack((np.ones(X1.size), X1, X2, X3))
    Y = Y.reshape(-1, 1)
    iterations = 100000
    learning_rate = 0.001
    w = train(X, Y, iterations, learning_rate)

    print("\nWeights = %s" % w.T)
    for i in range(5):
        print("X[%d] => %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))

    input("Press <Enter>")