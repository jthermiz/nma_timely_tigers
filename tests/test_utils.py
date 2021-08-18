from nma_timely_tigers import utils, models
import numpy as np
import os
import torch


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
    X, y = utils.toy_data(3)
    train_acc, epoch = utils.train(model, X, y)
    print(f'Train accuracy: {str(train_acc.values[epoch, 1])}')


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
    device = utils.set_device()
    X = utils.convert_to_tensor(X).to(device)
    y = utils.convert_to_tensor(y).to(device)
    utils.train(net, X, y)
    acc = utils.test(net, X, y)
    # Accuracy varies widely brain on network and training parameters!
    print(f'Train accuracy: {str(round(acc, 3))}')


def test_plot_accuracy():
    net = models.TwoLayer(2, 10, 3)
    X, y = utils.toy_data(3)
    acc_df, _ = utils.train(net, X, y)
    utils.plot_accuracy(acc_df)


def test_load_steinmetz():
    alldat = utils.load_steinmetz_dataset()
    print(f'Number of session: {str(len(alldat))}')


def test_shuffle():
    spks = np.repeat(np.arange(0, 10), 2, axis=0).reshape(10, -1)
    labels = np.zeros(10)
    labels[5:] = 1
    print('Original spikes and labels')
    print(spks, labels)
    print('')
    spks_shuf = utils.shuffle_neurons(spks, labels)
    print('Shuffled spikes')
    print(spks_shuf)


def test_calc_correlations():
    X = np.random.randn(10, 10)
    C = utils.calc_correlations(X)
    print(C)
    print(len(C))


def test_animal_correctness_labels():
    os.chdir('analyses/')
    alldat = utils.load_steinmetz_dataset()
    dat = alldat[0]
    z = utils.animal_correctness_labels(dat)
    print(z)
    print(len(z))
    os.chdir('..')


def test_transformer_simple():
    num_examples, in_dim, num_points = 32, 100, 75
    out_dim = 3
    X = torch.randn(num_examples, num_points, in_dim)
    net = models.Transformer(in_dim, out_dim)
    Y = net(X)
    print(Y.shape)


def test_transformer_train():
    num_examples, in_dim, num_points = 32, 100, 75
    out_dim = 3
    X = torch.randn(num_examples, num_points, in_dim)
    y = torch.randint(0, high=3, size=(num_examples, ))
    net = models.Transformer(in_dim, out_dim)
    criterion = torch.nn.functional.nll_loss
    acc, epoch = utils.train(net, X, y, criterion=None)
    print(f'Train accuracy: {str(round(acc, 3))}')
