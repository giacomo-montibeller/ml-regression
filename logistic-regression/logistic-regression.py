import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w):
    linear_combination = np.matmul(X, w)
    return sigmoid(linear_combination)

def classify(X, w):
    return np.round(predict(X, w))

def loss (X, Y, w):
    predictions = predict(X, w)
    first_term = Y * np.log(predictions)
    second_term = (1 - Y) * np.log(1 - predictions)
    return (-1) * np.average(first_term + second_term)

def gradient(X, Y, w):
    predictions = predict(X, w)
    return np.matmul(X.T, (predictions - Y)) / (2 * X.shape[0])

def train(X, Y, iterations, learning_rate):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iterations: %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * learning_rate
    return w

def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples

    print("\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))

if __name__ == "__main__":
    X1, X2, X3, Y = np.loadtxt("examples.txt", skiprows=1, unpack=True)
    X = np.column_stack((np.ones(X1.size), X1, X2, X3))
    Y = Y.reshape(-1, 1)
    iterations = 10000
    learning_rate = 0.001
    w = train(X, Y, iterations, learning_rate)

    test(X, Y, w)

    input("Press <Enter>")