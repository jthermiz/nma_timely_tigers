from nma_timely_tigers import utils, models
import numpy as np


def test_add():
    """Simple addition test
    """
    c = utils.add(1, 2)
    print(f'1 + 2 = {str(c)}')


def test_get_neurons_by_brain_area(dat):
    spks, ns = utils.get_neurons_by_brain_area(dat, 'VISa')
    print('Spike rate matrix')
    print(spks)
    print('Neuron indices')
    print(ns)


def test_toy_data():
    X, y = utils.toy_data(3)
    num_class = len(np.unique(y))
    for idx in range(num_class):
        mu = np.mean(X[y == idx])
        sigma = np.std(X[y == idx])
        print(f'Class: {str(idx)}')
        print(f'Total mean: {str(round(mu,3))}')
        print(f'Total std: {str(round(sigma,3))}')
        print('')


def test_model(model):
    X, y = utils.toy_data()
    train_acc = utils.train(model, X, y)  # TODO
    print(f'Train accuracy: {str(train_acc)}')


def test_shuffle_and_split_data():
    X, y = utils.toy_data(3)
    Xtrain, ytrain, Xtest, ytest = utils.shuffle_and_split_data(X, y)
    print(f'Xtrain shape: {str(Xtrain.shape)}')
    print(f'ytrain shape: {str(ytrain.shape)}')
    print(f'Xtest shape: {str(Xtest.shape)}')
    print(f'ytest shape: {str(ytest.shape)}')


def test_test():
    net = models.TwoLayer(2, 10, 3)
    X, y = utils.toy_data(3)
    porig = net.parameters()
    acc = utils.test(net, X, y)
    pnew = net.parameters()
    if not utils.parameters_equal(porig, pnew):
        print('Change in parameters! Fix test function!')
    print(f'Test accuracy: {str(round(acc, 3))}')


def test_train():
    net = models.TwoLayer(2, 10, 3)
    X, y = utils.toy_data(3)
    utils.train(net, X, y)
    acc = utils.test(net, X, y)
    # Accuracy varies widely brain on network and training parameters!
    print(f'Train accuracy: {str(round(acc, 3))}')


def test_plot_accuracy():
    net = models.TwoLayer(2, 10, 3)
    X, y = utils.toy_data(3)
    acc_df = utils.train(net, X, y)
    utils.plot_accuracy(acc_df)
