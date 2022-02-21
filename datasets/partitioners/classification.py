import torch
import numpy as np

def split_iid(data, n_clients, classes_per_client,
                shuffle=True, balancedness=1):
    """
    Splits data among `n_clients` s.t the labels are iid
    Args:
        data (List[Tuple]): pairs of sample specifier (eg. image file name) and labels 
        n_clients (int): number of clients
    Returns:
        List[Tuple]: specifier and client id 
    """
    # TODO: for now it just works for classification problems
    n_samples = len(data)
    sample_inds = np.array([d[0] for d in data])
    labels = np.array([d[1] for d in data])
    n_labels = np.max(labels) + 1

    if balancedness >= 1.0:
        data_per_client = [n_samples // n_clients] * n_clients
        data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients
    else:
        fracs = balancedness ** np.linspace(0, n_clients - 1, n_clients)
        fracs /= np.sum(fracs)
        fracs = 0.1 / n_clients + (1 - 0.1) * fracs
        data_per_client = [np.floor(frac * n_samples).astype('int') for frac in fracs]

        data_per_client = data_per_client[::-1]

        data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]

    assert sum(data_per_client) <= n_samples, print('sum(data_per_client) > n_samples. Impossible Split')

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []
        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(sample_inds[client_idcs], labels[client_idcs])]

    return clients_split
