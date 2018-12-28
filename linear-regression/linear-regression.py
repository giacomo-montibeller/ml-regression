import numpy as np

def predict(X, w, b):
    return X * w + b

def loss (X, Y, w, b):
    predictions = predict(X, w, b)
    return np.average((predictions - Y) ** 2)

def gradient(X, Y, w, b):
    predictions = predict(X, w, b)
    w_gradient = np.average(2 * X * (predictions - Y))
    b_gradient = np.average(2 * (predictions - Y))
    return (w_gradient, b_gradient)

def train_line(X, Y, iterations, learning_rate):
    w = b = 0;
    for i in range(iterations):
        print("%d => Loss: %.10f" % (i, loss(X, Y, w, b)))
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * learning_rate
        b -= b_gradient * learning_rate
    return w, b

if __name__ == "__main__":
    X, Y = np.loadtxt("examples.txt", skiprows=1, unpack=True)
    iterations = 20000
    learning_rate = 0.001
    w, b = train_line(X, Y, iterations, learning_rate)

    print("\nw = %.10f, b = %.10f" % (w, b))
    print("\ninput = %d => prediction = %.10f" % (25, predict(25, w, b)))

    input("Press <Enter>")

    import matplotlib.pyplot as plt
    
    plt.plot(X, Y, "bo")
    plt.xlabel("Input", fontsize=20)
    plt.ylabel("Output", fontsize=20)
    x_edge, y_edge = 50, 50
    plt.axis([0, x_edge, 0, y_edge])
    plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
    plt.show()