import torch
import torch.nn as nn
from torch.nn import utils
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt


def add(a, b):
    """Add two numbers

    Parameters
    ----------
    a : numeric
        Any number
    b : numeric
        Any number

    Returns
    -------
    Numeric
        Sum of a and b
    """
    return a + b


def toy_data(num_class, num_examples=100, num_features=2, noise=0.01):
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


def train(model, X, y, epochs=10, criterion=None, optimizer=None, **kwargs):
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
    Dataframe
        Train and validation accuracies
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=1e-2)

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

    X, y, Xval, yval = shuffle_and_split_data(X, y,
                                              train_frac=train_frac,
                                              seed=seed)

    X = convert_to_tensor(X).float()
    y = convert_to_tensor(y).long()
    train_data = TensorDataset(X, y)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    acc_mat = np.zeros((epochs, 2))
    model.train()
    for epoch in range(epochs):
        with tqdm(train_loader, unit='batch') as tepoch:
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

                train_acc = test(model, X, y)
                val_acc = test(model, Xval, yval)
                acc_mat[epoch, 0] = train_acc
                acc_mat[epoch, 1] = val_acc

    acc_df = pd.DataFrame(data=acc_mat,
                          columns=['Train', 'Test'],
                          index=range(epochs))
    return acc_df


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
    if type(X) == type(torch.Tensor()):
        return X
    else:
        return torch.Tensor(X)


def plot_accuracy(acc_df):
    acc_df.plot(xlabel='Epochs',
                ylabel='Accuracy',
                title='Training accuracy vs epoch')
    plt.show()
