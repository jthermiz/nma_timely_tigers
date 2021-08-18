from genericpath import exists
import torch
import torch.nn as nn
from torch.nn import utils
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests


def toy_data(num_class, num_examples=100, num_features=2, noise=0.01):
    """Generates toy dataset

    Parameters
    ----------
    num_class : Int
        Number of classes
    num_examples : int, optional
        Number of examples, by default 100
    num_features : int, optional
        Number of features, by default 2
    noise : float, optional
        Noise to signal ratio, by default 0.01

    Returns
    -------
    Tuple
        Feature matrix, labels
    """
    y = np.random.randint(num_class, size=num_examples)
    X = noise*np.random.randn(num_examples, num_features)
    for offset in range(num_class):
        X[y == offset] += offset
    return X, y


def set_device():
    """Set device

    Returns
    -------
    String
        Device name
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU")
    else:
        device = torch.device("cpu")
        print("using CPU")
    return device


def train(model, X, y, epochs=150, criterion=None, optimizer=None, **kwargs):
    """Train a model

    Parameters
    ----------
    model : nn.Module class
        pytorch model
    X : 2D array
        Feature matrix, examples by features
    y : 1D array
        Labels
    epochs : int, optional
        Number of training, by default 10
    criterion : loss class, optional
        Loss function, by default None which corresponds to cross entropy loss
    optimizer : optimizer class, optional
        Optimizer class, by default None which corresponds to SGD

    Returns
    -------
    Tuple
        Dataframe: Train and validation accuracies, 
        epoch: Epoch number where training stopped
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = optim.Adam(model.parameters())

    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']
    else:
        batch_size = 32

    if 'device' in kwargs:
        device = kwargs['device']
    else:
        device = set_device()

    if 'train_frac' in kwargs:
        train_frac = kwargs['train_frac']
    else:
        train_frac = 0.7

    if 'seed' in kwargs:
        seed = kwargs['seed']
    else:
        seed = 0

    if 'early_stop_count' in kwargs:
        early_stop_count = kwargs['early_stop_count']
    else:
        early_stop_count = epochs+1

    X = convert_to_tensor(X).float()
    y = convert_to_tensor(y).long()

    X = X.to(device)
    y = y.to(device)

    X, y, Xval, yval = shuffle_and_split_data(X, y,
                                              train_frac=train_frac,
                                              seed=seed)

    train_data = TensorDataset(X, y)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    acc_mat = np.zeros((epochs, 2))
    model.to(device)
    model.train()
    val_best_acc = 0
    early_stop_ctr = 0
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            #loss = criterion(torch.argmax(output, dim=1), target)
            # errors out during transformer training
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_acc = test(model, X, y)
            val_acc = test(model, Xval, yval)
            acc_mat[epoch, 0] = train_acc
            acc_mat[epoch, 1] = val_acc

        if val_acc < val_best_acc:
            early_stop_ctr += 1
        else:
            val_best_acc = val_acc
            early_stop_ctr = 0

        if early_stop_ctr == early_stop_count:
            print(f'Early stopped at {str(epoch)}')
            break

        if (epoch % 1000) == 0:
            print(f'On epoch {str(epoch)} of {str(epochs)}')
            print(f'Train accuracy: {str(round(train_acc, 3))}')
            print(f'Validation accuracy: {str(round(val_acc, 3))}')
            print('')

    print('Finished training!')
    acc_df = pd.DataFrame(data=acc_mat,
                          columns=['Train', 'Validation'],
                          index=range(epochs))
    return acc_df, epoch


def test(model, X, y, **kwargs):
    """Test model

    Parameters
    ----------
    model : pytorch model
        Neural network
    X : 2D array
        Feature matrix, examples by features
    y : 1D array
        Labels

    Returns
    -------
    Numeric
        Test accuracy
    """
    correct = 0
    total = 0

    X = convert_to_tensor(X)
    y = convert_to_tensor(y)
    total = len(y)
    data_loader = TensorDataset(X, y)
    for data in data_loader:
        inputs, labels = data
        # Assume that everything is in the same memory space
        #inputs = inputs.to(device).float()
        #labels = labels.to(device).long()

        outputs = model(inputs)
        predicted = torch.argmax(outputs)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc


def shuffle_and_split_data(X, y, train_frac=0.7, seed=1):
    """Shuffle dataset into train and test set

    Parameters
    ----------
    X : 2D array
        Feature matrix (examples by features)
    y : 1D array
        Label array
    train_frac : float, optional
        Fraction of samples to split to train set, by default 0.7
    seed : int, optional
        Random seed, by default 1

    Returns
    -------
    tuple
        Train features and labels and test features and labels
    """
    # set seed for reproducibility
    np.random.seed(seed)
    # Number of samples
    N = X.shape[0]
    # Shuffle data
    # get indices to shuffle data, could use torch.randperm
    shuffled_indices = np.random.randint(N, size=N)
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    # Split data into train/test
    # assign test datset size using 20% of samples
    train_size = int(train_frac * N)
    X_test = X[:train_size]
    y_test = y[:train_size]
    X_train = X[train_size:]
    y_train = y[train_size:]

    return X_test, y_test, X_train, y_train


def average_trials_across_time(spks, start_time, end_time, fs=100):
    """Average trials within a specified time window

    Parameters
    ----------
    spks : 3D array
        Matrix of spike rates (trials x neurons x time samples). Spike matrix (eg dat['spks])
    start_time : Numeric
        Start time in seconds relative to t=0
    end_time : Numeric
        End time in seconds
    fs : Numeric, optional
        Sample rate of binned spikes in hertz, by default 100

    Returns
    -------
    2D array
        Feature matrix trials x neurons
    """
    start_frame, end_frame = int(start_time*fs), int(end_time*fs)
    features = spks[:, :, start_frame:end_frame].mean(axis=2)
    return features


def get_neurons_by_brain_area(dat, areas=[]):
    """Get spikes from specified area(s)

    Parameters
    ----------
    dat : dict
        Steinmetz dictionary for storing session data
    areas : list, optional
        Brain areas, by default []. List of all brain areas:
        [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL",
                    "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB",
                    "ORBm", "PIR", "PL", "SSp", "SSs", "RSP","TT"], # non-visual cortex
                ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN",
                    "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                ["ACB", "CP", "GPe", "LS", "LSc", "LSr",
                    "MS", "OT", "SNr", "SI"], # basal ganglia
                ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                ]

    Returns
    -------
    [type]
        [description]
    """
    neurons = []
    num_neurons = dat['spks'].shape[0]
    for nidx in range(num_neurons):
        if dat['brain_area'][nidx] in areas:
            neurons.append(nidx)
    area_spks = dat['spks'][neurons]
    return area_spks, neurons


def parameters_equal(param1, param2):
    """Checks to see if model parameters from two models are equal

    Parameters
    ----------
    param1 : Generator
        Generator from nn.Module.parameters()
    param2 : Generator
        Generator from nn.Module.parameters()

    Returns
    -------
    bool
        Whether parameters are equal
    """
    for ele1 in param1:
        ele2 = next(param2)
        if not torch.equal(ele1, ele2):
            return False
    return True


def convert_to_tensor(X):
    """Convert numpy to tensor

    Parameters
    ----------
    X : Numpy (or Tensor)
        ND array

    Returns
    -------
    Tensor
        Tensor
    """
    if type(X) == type(torch.Tensor()):
        return X
    else:
        return torch.Tensor(X)


def plot_accuracy(acc_df):
    """Plots accuracy training curve

    Parameters
    ----------
    acc_df : DataFrame
        2-column dataframe with train and validation accuracy
    """
    acc_df.plot(xlabel='Epochs',
                ylabel='Accuracy',
                title='Training accuracy vs epoch')
    plt.show()
    return plt.gcf, plt.gca


def load_steinmetz_dataset():
    """Loads Steinmetz dataset

    Returns
    -------
    List
        List of 'dat' dictionary for each experimental session
    """

    if not exists('steinmetz_part0.npz'):
        print('Attempting to download data')
        fname = []
        for j in range(3):
            fname.append('steinmetz_part%d.npz' % j)
            url = ["https://osf.io/agvxh/download"]
            url.append("https://osf.io/uv3mw/download")
            url.append("https://osf.io/ehmw2/download")

        for j in range(len(url)):
            if not os.path.isfile(fname[j]):
                try:
                    r = requests.get(url[j])
                except requests.ConnectionError:
                    print("!!! Failed to download data !!!")
                else:
                    if r.status_code != requests.codes.ok:
                        print("!!! Failed to download data !!!")
                    else:
                        with open(fname[j], "wb") as fid:
                            fid.write(r.content)
    else:
        print('Loading data from disk')
        fname = ['steinmetz_part%d.npz' % j for j in range(3)]

    alldat = np.array([])
    for j in range(len(fname)):
        alldat = np.hstack(
            (alldat, np.load('steinmetz_part%d.npz' % j, allow_pickle=True)['dat']))

    return alldat


def shuffle_neurons(spks, labels, seed=0):
    """Shuffle neurons across same trials

    Parameters
    ----------
    spks : numpy.array
        Spike matrix, trials x neurons
    labels : sequence
        Sequence of trial labels
    seed : int, optional
        Random seed, by default 0

    Returns
    -------
    [type]
        [description]
    """
    types = np.unique(labels)
    spks_shuf = np.zeros_like(spks)

    for type in types:
        tmp = spks[type == labels]
        num_trial, num_neuron = tmp.shape
        for nidx in range(num_neuron):
            np.random.seed(nidx)
            shuf_idx = np.random.permutation(num_trial)
            spks_shuf[type == labels, nidx] = tmp[shuf_idx, nidx]

    return spks_shuf


def norm_l1(model):
    """Calculate l1 norm

    Parameters
    ----------
    model : nn.module
        Pytorch

    Returns
    -------
    Tensor
        l1 morm
    """
    norm = 0
    for param in model.parameters():
        norm += torch.abs(torch.flatten(param))
    return norm


def stimulus_labels(dat):
    """Stimulus lables for Steinmetz animal experiment

    Parameters
    ----------
    dat : dict
        Steinmetz dat dict

    Returns
    -------
    1D array
        Stimulus labels
    """
    num_trials = dat['spks'].shape[1]
    y = np.zeros(num_trials)

    left = dat['contrast_left']
    right = dat['contrast_right']

    y[left > right] = 1
    y[left < right] = 2

    return y


def calc_correlations(X):
    """Calculate correlation pairs among all columns

    Parameters
    ----------
    X : 2D array
        Feature matrix, examples by variables

    Returns
    -------
    1D array
        Correlations with everything at or above the diagnonal removed
    """
    C = np.corrcoef(X)
    C = np.tril(C)
    C = C[C != 0]
    C = C[C != 1]
    C = np.reshape(C, (-1, 1))
    return C


def _remap_label(x):
    """Remap stimulus label

    Parameters
    ----------
    x : Int
        Stimulus label

    Returns
    -------
    Int
        Stimulus label
    """
    if x == 0:
        x = 1
    elif x == 1:
        x = 2
    elif x == 2:
        x = 0
    return x


def animal_correctness_labels(dat):
    """Whether the animal made the correct or incorrect choice

    Parameters
    ----------
    dat : dict
        Steinmetz dat dict

    Returns
    -------
    1D array
        The animal correctness labels
    """
    y = stimulus_labels(dat)  # 0: nogo, 1: left, 2: right
    y = list(map(_remap_label, y))  # 0: right, 1: nogo, 2: left
    yh = dat['response'] + 1  # 0: right, 1: nogo, 2: left
    z = np.zeros(len(y))  # incorrect
    z[yh == y] = 1  # correct
    return z
