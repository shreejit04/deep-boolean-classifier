import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return np.tanh(x)


def predict(x, hidden_weights, output_weights):
    hidden_activations = sigmoid(np.dot(x, hidden_weights))
    #print(hidden_activations)
    output = np.dot(hidden_activations, output_weights)
    return np.sign(output)


def train_neural_network(x_train, y_train, num_hidden_units, learning_rate):
    dim = len(x_train[0])
    hidden_weights = np.random.rand(dim, num_hidden_units)
    output_weights = np.random.rand(num_hidden_units, 1)
    status = False
    attempt = 0
    errors = []
    attempts = []

    while not status:
        attempt += 1
        error_count = 0
        for index in range(len(x_train)):
            hidden_activations = sigmoid(np.dot(x_train[index], hidden_weights))
            output = np.dot(hidden_activations, output_weights)

            error = 0.5*(y_train[index] - np.sign(output))**2
            output_weights += learning_rate * error * hidden_activations
            hidden_weights += learning_rate * error * np.outer(x_train[index], hidden_activations * (1 - hidden_activations)) @ output_weights.T

        predictions = [predict(x, hidden_weights, output_weights) for x in x_train]

        for i in range(len(predictions)):
            if y_train[i] != predictions[i]:
                error_count += 1

        errors.append(error_count)
        attempts.append(attempt)
        print(error_count)

        plt.plot(attempts, errors)
        plt.xlabel("attempt")
        plt.ylabel("errors")
        plt.savefig('see.png')

        if error_count <= 120:
            break

    return hidden_weights, output_weights


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            data_point = [float(value) for value in line.strip().split()]
            data.append(data_point)
    return np.array(data)


def load_labels(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        values = content[1:-1].split(', ')
        labels = [float(value) for value in values]
    return np.array(labels)


def main():
    # Load training data and labels
    x_train = load_data('a2-train-data.txt')
    y_train = load_labels('a2-train-label.txt')

    # Set hyperparameters
    num_hidden_units = 1
    learning_rate = 0.04
    status = False


    # Train the neural network
    hidden_weights, output_weights = train_neural_network(x_train, y_train, num_hidden_units, learning_rate)

    # Save the model
    with open('model.txt', 'w') as model_file:
        model_file.write(str(num_hidden_units) + '\n')
        model_file.write(' '.join(map(str, output_weights)) + '\n')
        for weights in hidden_weights:
            model_file.write(' '.join(map(str, weights)) + '\n')

    # Load testing data and labels
    x_test = load_data('a2-test-data.txt')
    y_test = load_labels('a2-test-label.txt')

    # Generate predictions on the testing set
    predictions = [predict(x, hidden_weights, output_weights) for x in x_test]
    with open("predictions.txt", 'w') as predictions_file:
        predictions_file.write(' '.join(map(str, predictions)))

    error_count = 0

    for i in range(len(predictions)):
        if y_train[i] != predictions[i]:
            error_count += 1


if __name__ == "__main__":
    main()
