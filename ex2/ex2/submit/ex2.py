import sys
import random
from threading import Thread
import numpy as np
from numpy import linalg as LA
from scipy import stats
from random import randint


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def validate_PA(validation_x, validation_y, PA_w):
    PA_w_c = PA_w.copy()
    miss_counter_validation = 0
    for x, y in zip(validation_x, validation_y):
        y_hat_validation = np.argmax(np.dot(PA_w_c, x))
        if y != int(y_hat_validation):
            miss_counter_validation += 1
            '''
            loss = PA_loss(x, y, y_hat_validation, PA_w_c)
            eta = PA_eta(loss, x)
            # updating the vectors
            PA_w_c[int(y)] = PA_w_c[int(y)] + eta * x
            PA_w_c[int(y_hat_validation)] = PA_w_c[int(y_hat_validation)] - eta * x
            '''
    return miss_counter_validation / len(validation_x)


def switch(argument):
    # turns sex to points
    switcher = {
        'M': [1.0, 0.0, 0.0],
        'F': [0.0, 1.0, 0.0],
        'I': [0.0, 0.0, 1.0],
    }
    return switcher.get(argument)


def k_delta(y, argmax_no_delta):
    argmax_no_delta = np.asarray(argmax_no_delta).astype(float)
    for i in range(len(argmax_no_delta)):
        if i != int(y):
            argmax_no_delta[i] += 1
    return argmax_no_delta


def delt(y, y_hat_tag):
    if str(int(y)) == str(int(y_hat_tag)):
        return 0
    else:
        return 1


def validate_perceptron(validation_x, validation_y, Perceptron_w):
    validation_miss_counter = 0
    for x, y in zip(validation_x, validation_y):
        y_hat = np.argmax(np.dot(Perceptron_w, x))
        if y != y_hat:
            validation_miss_counter += 1
    return validation_miss_counter / len(validation_x)


def Perceptron(eta, epochs, depth_counter, Perceptron_lines):
    # three rows for the vectors first is a coordinate for [M, F, I]
    Perceptron_w = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    Perceptron_eta = eta

    # shuffling the examples
    lines = Perceptron_lines
    random.shuffle(lines)

    X = []
    Y = []

    for line in lines:
        x = list(line.strip().split(','))
        y = float(x.pop())
        x = np.asarray(x)
        x = x.astype(float)
        X.append(x)
        Y.append(y)

    X = np.asarray(X)
    stats.zscore(X)
    Y = np.asarray(Y)

    train_x, validation_x = X[:int(len(X) * .8)], X[int(len(X) * .8):]
    train_y, validation_y = Y[:int(len(Y) * .8)], Y[int(len(Y) * .8):]

    counter = 0
    train_miss_rate = 0
    overall_train = 0

    for epoch in range(epochs):

        # reading one example from the file
        for i in range(len(train_x)):
            overall_train += 1
            counter += 1
            j = (randint(0, len(train_x) - 1))
            x = train_x[j]
            y = train_y[j]
            y_hat = np.argmax(np.dot(Perceptron_w, x))
            if y != y_hat:
                # updating the vectors
                train_miss_rate += 1
                Perceptron_w[int(y)] = Perceptron_w[int(y)] + Perceptron_eta * x
                Perceptron_w[int(y_hat)] = Perceptron_w[int(y_hat)] - Perceptron_eta * x
    if validate_perceptron(validation_x, validation_y, Perceptron_w) > 0.38 and depth_counter < 50:
        depth_counter += 1
        Perceptron_w = Perceptron(eta, epochs, depth_counter, Perceptron_lines)
    return Perceptron_w


def validate_SVM(validation_x, validation_y, SVM_w):
    validate_miss_counter = 0

    # SVM test : reading one example from the file
    for x, y in zip(validation_x, validation_y):
        y_hat_tag = np.argmax(np.dot(SVM_w, x))
        if (int(y != y_hat_tag) - np.dot(SVM_w[int(y)], x) + np.dot(SVM_w[int(y_hat_tag)], x)) > 0:
            validate_miss_counter += 1
    return validate_miss_counter / len(validation_x)


def SVM(eta, epochs, SVM_lines):
    # three rows for the vectors first is a coordinate for [M, F, I]
    SVM_w = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    SVM_w = np.asarray(SVM_w)

    SVM_eta = eta

    # shuffling the examples
    lines = SVM_lines
    random.shuffle(lines)

    X = []
    Y = []

    for line in lines:
        x = list(line.strip().split(','))
        y = float(x.pop())
        x = np.asarray(x)
        x = x.astype(float)
        X.append(x)
        Y.append(y)

    X = np.asarray(X)
    stats.zscore(X)
    Y = np.asarray(Y)

    train_x, validation_x = X[:int(len(X) * .8)], X[int(len(X) * .8):]
    train_y, validation_y = Y[:int(len(Y) * .8)], Y[int(len(Y) * .8):]

    counter = 0
    train_miss_rate = 0
    overall_train = 0

    lowest_validate_miss = 1
    lowest_train_miss = 1
    optimised_w = SVM_w

    for epoch in range(epochs):
        lam = 1 / (epoch + 1)
        for i in range(len(train_x)):
            overall_train += 1
            counter += 1
            j = (randint(0, len(train_x) - 1))
            x = train_x[j]
            y = train_y[j]
            y_hat_tag = np.argmax(k_delta(y, np.dot(SVM_w, x)))
            if (int(delt(y, y_hat_tag)) - np.dot(SVM_w[int(y)], x) + np.dot(SVM_w[int(y_hat_tag)], x)) > 0:
                train_miss_rate += 1
                # updating the vectors
                SVM_w[int(y)] = (1 - SVM_eta * lam) * SVM_w[int(y)] + SVM_eta * x
                SVM_w[int(y_hat_tag)] = (1 - SVM_eta * lam) * SVM_w[int(y_hat_tag)] - SVM_eta * x
                for other_y in range(len(SVM_w)):
                    if other_y != y and other_y != y_hat_tag:
                        SVM_w[int(other_y)] = (1 - SVM_eta * lam) * SVM_w[int(other_y)]
            if counter > len(train_x) / 1:
                counter = 0
                val_miss = validate_SVM(validation_x, validation_y, SVM_w)
                t_miss = (train_miss_rate / overall_train)
                if val_miss < lowest_validate_miss and t_miss < lowest_train_miss:
                    # print("train miss: " + str(t_miss) + " validation miss: " + str(val_miss))
                    lowest_validate_miss = val_miss
                    lowest_train_miss = t_miss
                    optimised_w = SVM_w
                else:
                    SVM_w = optimised_w
    return optimised_w


def PA_loss(x_t, y_t, y_hat, w):
    x_t = np.asarray(x_t)
    y_t = int(y_t)
    y_hat = int(y_hat)
    w = np.asarray(w)
    return np.maximum(0, 1 - float(np.dot(w[y_t], x_t) + np.dot(w[y_hat], x_t)))


def PA_eta(loss, x):
    ret = loss / (2 * LA.norm(np.asarray(x)))
    if isinstance(ret, float):
        return ret
    else:
        return 0.0


def PA(epochs, depth_counter, PA_lines):
    # three rows for the vectors first is a coordinate for [M, F, I]
    PA_w = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    PA_w = np.asarray(PA_w)

    # shuffling the examples
    lines = PA_lines
    random.shuffle(lines)

    X = []
    Y = []

    for line in lines:
        x = list(line.strip().split(','))
        y = float(x.pop())
        x = np.asarray(x)
        x = x.astype(float)
        X.append(x)
        Y.append(y)

    X = np.asarray(X)
    stats.zscore(X)
    Y = np.asarray(Y)

    train_x, validation_x = X[:int(len(X) * .8)], X[int(len(X) * .8):]
    train_y, validation_y = Y[:int(len(Y) * .8)], Y[int(len(Y) * .8):]

    counter = 0
    train_miss_rate = 0
    overall_train = 0

    b = False
    lowest_validate_miss = 1
    lowest_train_miss = 1
    optimised_w = PA_w
    for epoch in range(epochs):

        # reading one example from the file
        for i in range(len(train_x)):
            overall_train += 1
            counter += 1
            j = (randint(0, len(train_x) - 1))
            x = train_x[j]
            y = train_y[j]
            y_hat = np.argmax(np.dot(PA_w, x))
            if y != int(y_hat):
                train_miss_rate += 1
                loss = PA_loss(x, y, y_hat, PA_w)
                eta = PA_eta(loss, x)
                # updating the vectors
                PA_w[int(y)] = PA_w[int(y)] + eta * x
                PA_w[int(y_hat)] = PA_w[int(y_hat)] - eta * x
            if counter > len(train_x) / 5:
                counter = 0
                val_miss = validate_PA(validation_x, validation_y, PA_w)
                t_miss = (train_miss_rate / overall_train)
                if val_miss < lowest_validate_miss and t_miss < lowest_train_miss:
                    lowest_validate_miss = val_miss
                    lowest_train_miss = t_miss
                    optimised_w = PA_w
                else:
                    PA_w = optimised_w

    if validate_PA(validation_x, validation_y, PA_w) > 0.38 and depth_counter < 50:
        optimised_w = PA(epochs, (depth_counter + 1), PA_lines)
    return optimised_w


''' MAIN: '''
train_x = open(sys.argv[1], "r")
train_y = open(sys.argv[2], "r")
test_x = open(sys.argv[3], "r")

t_perceptron = ""
t_SVM = ""
t_PA = ""
best_counter = 0

# creating new file with the x and y together
for line_x, line_y in zip(train_x.readlines(), train_y.readlines()):
    x = list(line_x.strip().split(','))
    g = switch(x[0])
    t_perceptron += str(g[0]) + ',' + str(g[1]) + ',' + str(g[2]) + line_x[1:].strip() + "," + line_y
    t_SVM += str(g[0]) + ',' + str(g[1]) + ',' + str(g[2]) + line_x[1:].strip() + "," + line_y
    t_PA += str(g[0]) + ',' + str(g[1]) + ',' + str(g[2]) + line_x[1:].strip() + "," + line_y
    best_counter += 1

for i in range(1):  # times to run with the same division to test and train

    # PA creation
    pa = ThreadWithReturnValue(target=PA, args=(20, 0, t_PA.split(),))
    pa.start()

    # SVM creation
    s = ThreadWithReturnValue(target=SVM, args=(0.001, 1500, t_SVM.split(),))
    s.start()

    # Perceptron creation
    ps = ThreadWithReturnValue(target=Perceptron, args=(0.001, 20, 0, t_perceptron.split(),))
    ps.start()

    # getting the matrixes
    Perceptron_w = ps.join()
    Perceptron_w = np.asarray(Perceptron_w)
    PA_w = pa.join()
    PA_w = np.asarray(PA_w)
    SVM_w = s.join()
    SVM_w = np.asarray(SVM_w)

    # Perceptron test : reading one example from the file
    for line_x in test_x.readlines():
        x = list(line_x.strip().split(','))
        x = np.concatenate((switch(x.pop(0)), x), axis=0)
        x = x.astype(float)
        y_hat_perceptron = int(np.argmax(np.dot(Perceptron_w, x)))
        y_hat_SVM = int(np.argmax(np.dot(SVM_w, x)))
        y_hat_PA = int(np.argmax(np.dot(PA_w, x)))
        print("perceptron: " + str(int(y_hat_perceptron)) + ", svm: "
              + str(int(y_hat_SVM)) + ", pa: " + str(int(y_hat_PA)))
