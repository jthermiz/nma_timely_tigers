from nma_timely_tigers import utils
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


def test_model(model):
    N = 100
    num_feat = 2
    noise = 0.01
    y = np.random.randint(1, size=N)
    X = noise*np.random.randn(N, num_feat)
    X[y == 1] += 1
    train_acc = utils.train(model, X, y)  # TODO
    print(f'Train accuracy: {str(train_acc)}')
