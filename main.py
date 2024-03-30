import matplotlib.pyplot as plt

x = [0.2, 0.4, 0.6, 0.8]
y = [0.1, 0.3, 0.5, 0.7]

def backtrack(w, b, eta, epoch):
    losses = []
    for i in range(epoch):
        loss = 0
        for j in range(len(x)):
            # Forward pass
            z = w * x[j] + b
            a = 1 / (1 + 2.71828 ** (-z))
            # Backward pass
            dz = a - y[j]
            dw = x[j] * dz
            db = dz
            # Update weights
            w = w - eta * dw
            b = b - eta * db
            loss += (a - y[j]) ** 2
        losses.append(loss)
        if i % 10 == 0:
            print(f"Epoch: {i+1}, Loss: {loss}")
    print(f"Final w: {w}, Final b: {b}")
    return w, b

def test(w, b):
    for i in range(len(x)):
        z = w * x[i] + b
        a = 1 / (1 + 2.71828 ** (-z))
        print(f"Prediction: {a}, True: {y[i]}")

def predict(w, b, x):
    z = w * x + b
    a = 1 / (1 + 2.71828 ** (-z))
    return a


if __name__ == "__main__":
    w, b = -2, 1
    plt.scatter(x, y, color='blue')
    plt.plot(x, [predict(w, b, i) for i in x], color='green')
    w, b = backtrack(w, b, 1, 1000)
    test(w, b)
    plt.plot(x, [predict(w, b, i) for i in x], color='red')
    plt.show()