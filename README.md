# deep-boolean-classifier

Implements a Boolean classifier using a neural network with one output and a single hidden layer.

•	The code uses a training set with 1000 dimensions, provided in gzipped format, including training data set and labels. The algorithm is able to obtain a decent level of accuracy on the training set with a focus on minimizing misclassifications.

Then, it applies the trained classifier to a testing set (gzipped) comprising testing data set and labels and quantifies the number of misclassifications made by the neural network on the testing set.

As output, the program creates a text file consolidating Boolean predictions of the classifier on the testing set as a sequence of +1 and -1, separated by spaces in a single line.

•	Generates a text file encapsulating the neural network configuration.

•	Initial line: Number of hidden units.

•	Sequence of weights for the output unit.

•	Weights assigned to each hidden unit concerning the inputs.

•	M+2 lines in total, with the final M lines each containing 1000 entries.
