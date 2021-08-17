import numpy as np

def gradient(w, x, y):
    print(x.shape)
    y_estimate = x.dot(w).flatten()
    y_error = y.flatten() - y_estimate
    gradient = -(1/len(x))*y_error.dot(x)
    return gradient, np.power(y_error, 2)

if __name__ == '__main__':
    data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
    data_y = np.sin(data_x) + 0.1 * np.power(data_x, 2) + 0.5 * np.random.randn(100, 1)
    data_x /= np.max(data_x)

    # In order to simplify our model we include the intercept in the input values,
    # Thus, we dont have to carry the bias (b) term through the calculation,
    # that is done by adding a column of ones to the data, this way our model becomes simply y=wTx.
    data_x = np.hstack((np.ones_like(data_x), data_x))
    # print(data_x)
    # seperate to train and test data
    order = np.random.permutation(len(data_x))
    portion = 20
    test_x = data_x[order[:portion]]
    test_y = data_y[order[:portion]]
    train_x = data_x[order[portion:]]
    train_y = data_y[order[portion:]]

    # Gradient Descent
    w = np.random.randn(2)
    print(w.shape)


    alpha = 0.5
    stopping_criteria = 1e-5

    iterations = 1
    while True:
        gradients, error = gradient(w, train_x, train_y)
        new_w = w - (alpha*gradients)

        # converge when our model stops changing significantly
        if np.sum(abs(new_w - w)) < stopping_criteria:
            print ("converged")
            break

        if iterations % 100 == 0:
            print "Iteration: %d - Error: %.4f" % (iterations, error)

        iterations += 1
        w = new_w