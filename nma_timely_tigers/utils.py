import torch
import torch.nn as nn
import torch.optim as optim


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


def train(model, X, y, **kwargs):
    NotImplementedError('TODO')


def test(model, X, y, **kwargs):
    NotImplementedError('TODO')


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
                ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP","TT"], # non-visual cortex
                ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
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
