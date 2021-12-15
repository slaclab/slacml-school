import numpy as np
from matplotlib import pyplot as plt

def plot_confusion_matrix(labels, predictions, normalize=False):
    plt.figure(figsize=(8,8))
    n_classes = len(np.unique(labels))
    hist, xbins, ybins, im = plt.hist2d(labels, predictions, range=[[-0.5,n_classes-0.5],[-0.5,n_classes-0.5]], bins=[n_classes,n_classes])
    norms = np.sum(hist, axis=0) if normalize else np.ones(n_classes)
    plt.pcolormesh(xbins, ybins, hist/norms, cmap=plt.cm.Blues)
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            plt.text(xbins[j]+0.5,ybins[i]+0.5, 
                     '{:0.3f}'.format(hist[i,j]/norms[j]) if normalize else '{:0.0f}'.format(hist[i,j]), 
                     ha="center", va="center")
    plt.xlabel('True class')
    plt.ylabel('Predicted class')
    plt.show()
    

def plot_softmax(labels,softmax,class_names=None):
    import numpy as np
    import matplotlib.pyplot as plt
    num_class   = len(softmax[0])
    if class_names is None:
        class_names = np.arange(num_class)
    assert num_class == len(class_names)
    unit_angle  = 2*np.pi/num_class
    xs = np.array([ np.sin(unit_angle*i) for i in range(num_class+1)])
    ys = np.array([ np.cos(unit_angle*i) for i in range(num_class+1)])
    fig,axis=plt.subplots(figsize=(8,8),facecolor='w')
    plt.plot(xs,ys)#,linestyle='',marker='o',markersize=20)
    for d,name in enumerate(class_names):
        plt.text(xs[d]*1.1,ys[d]*1.1,str(name),fontsize=24,ha='center',va='center',rotation=-unit_angle*d/np.pi*180)
    plt.xlim(-1.3,1.3)
    plt.ylim(-1.3,1.3)
    xs=xs[0:num_class]
    ys=ys[0:num_class]
    for label in range(num_class):
        idx = np.where(labels==label)
        scores=softmax[idx]
        xpos=[np.sum(xs * s) for s in scores]
        ypos=[np.sum(ys * s) for s in scores]
        plt.plot(xpos,ypos,linestyle='',marker='o',markersize=10,alpha=0.5)
    axis.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelleft=False,labelbottom=False)
    plt.show()
    
def contingency_table(a, b, na=None, nb=None):
    if not na:
        na = np.max(a)
    if not nb:
        nb = np.max(b)
    table = np.zeros((na, nb), dtype=np.int)
    for i, j in zip(a,b):
        table[i,j] += 1
    return table

def purity(truth, pred):
    pred, pcts = np.unique(pred, return_counts=True, return_inverse=True)[1:]
    truth, tcts = np.unique(truth, return_counts=True, return_inverse=True)[1:]
    table = contingency_table(pred, truth, len(pcts), len(tcts))
    purities = table.max(axis=1) / pcts
    return purities.mean()

def efficiency(truth, pred):
    pred, pcts = np.unique(pred, return_counts=True, return_inverse=True)[1:]
    truth, tcts = np.unique(truth, return_counts=True, return_inverse=True)[1:]
    table = contingency_table(pred, truth, len(pcts), len(tcts))
    efficiencies = table.max(axis=0) / tcts
    return efficiencies.mean()

def union_find(new_edges, n, group_ids=None, groups=None):
    """
    Implementation of the Union-Find algorithm.

    Args:
        new_edges (np.ndarray): (E,2) Additional edges
        n (int)               : Total number of clusters C
        group_ids (array)     : (C) List of group ids
        groups (dict of array): Dictionary of partition groups

    Returns:
        np.ndarray: (C) Updated list of group ids
        np.ndarray: (C) Updated dictionary of partition groups
    """
    if group_ids is None:
        group_ids = np.arange(n)
    if groups is None:
        groups = {}
    for i, j in new_edges:
        leaderi = group_ids[i]
        leaderj = group_ids[j]
        if leaderi in groups:
            if leaderj in groups:
                if leaderi == leaderj: continue # nothing to do
                groupi = groups[leaderi]
                groupj = groups[leaderj]
                if len(groupi) < len(groupj):
                    i, leaderi, groupi, j, leaderj, groupj = j, leaderj, groupj, i, leaderi, groupi
                groupi |= groupj
                del groups[leaderj]
                for k in groupj:
                    group_ids[k] = leaderi
            else:
                groups[leaderi].add(j)
                group_ids[j] = leaderi
        else:
            if leaderj in groups:
                groups[leaderj].add(i)
                group_ids[i] = leaderj
            else:
                group_ids[i] = group_ids[j] = i
                groups[i] = set([i, j])

    return group_ids, groups


def node_assignment(edge_index, edge_label, n):
    """
    Function that assigns each node to a group, based
    on the edge assigment provided. This uses a local
    union find implementation.

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix
        edge_assn (np.ndarray) : (E) Boolean array (1 if edge is on)
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (C) List of group ids
    """
    # Loop over on edges, reset the group IDs of connected node
    on_edges = edge_index[np.where(edge_label)[0]]
    return union_find(on_edges, n)[0]

def adjacency_matrix(edge_index, n):
    """
    Function that creates an adjacency matrix from a list
    of connected edges in a graph.

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (C,C) Adjacency matrix
    """
    adj_mat = np.eye(n)
    adj_mat[tuple(edge_index.T)] = 1
    return adj_mat


def grouping_adjacency_matrix(edge_index, adj_mat):
    """
    Function that creates an adjacency matrix from a list
    of connected edges in a graph by considering nodes to
    be adjacent when they belong to the same group, provided
    that they are connected in the input graph.

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix selected from adjacency matrix
        ajd_mat (np.ndarray)   : (C,C) Input adjacency matrix
    Returns:
        np.ndarray: (C,C) Adjacency matrix
    """
    n = adj_mat.shape[0]
    node_assn = node_assignment(edge_index, np.ones(len(edge_index)), n)
    return adj_mat*(node_assn == node_assn.reshape(-1,1))


def grouping_loss(pred_mat, edge_index, adj_mat, loss='ce'):
    """
    Function that defines the graph clustering score.
    Given a target adjacency matrix A as and a predicted
    adjacency P, the score is evaluated the average
    L1 or L2 distance between truth and prediction.

    Args:
        pred_mat (np.ndarray)  : (C,C) Predicted adjacency matrix
        edge_index (np.ndarray): (E,2) Incidence matrix
        ajd_mat (np.ndarray)   : (C,C) Input adjacency matrix
    Returns:
        int: Graph grouping loss
    """
    adj_mat = grouping_adjacency_matrix(edge_index, adj_mat)
    if loss == 'ce':
        from sklearn.metrics import log_loss
        return log_loss(adj_mat.reshape(-1), pred_mat.reshape(-1), labels=[0,1])
    elif loss == 'l1':
        return np.mean(np.absolute(pred_mat-adj_mat))
    elif loss == 'l2':
        return np.mean((pred_mat-adj_mat)*(pred_mat-adj_mat))
    else:
        raise ValueError('Loss type not recognized: {}'.format(loss))


def edge_assignment_score(edge_index, edge_scores, n):
    """
    Function that finds the graph that produces the lowest
    grouping score iteratively adding the most likely edges,
    if they improve the the score (builds a spanning tree).

    Args:
        edge_index (np.ndarray) : (E,2) Incidence matrix
        edge_scores (np.ndarray): (E,2) Two-channel edge score
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (E',2) Optimal incidence matrix
        float     : Score for the optimal incidence matrix
    """
    # Build an input adjacency matrix to constrain the edge selection to the input graph
    adj_mat = adjacency_matrix(edge_index, n)

    # Interpret the score as a distance matrix, build an MST based on score
    from scipy.special import softmax
    edge_scores = softmax(edge_scores, axis=1)
    pred_mat = np.eye(n)
    pred_mat[tuple(edge_index.T)] = edge_scores[:,1]

    # Order the edges by increasing order of OFF score
    args = np.argsort(edge_scores[:,0])
    ord_index = edge_index[args]

    # Now iteratively add edges, until the total score cannot be improved any longer
    best_index = np.empty((0,2), dtype=int)
    best_groups = np.arange(n)
    best_loss = grouping_loss(pred_mat, best_index, adj_mat)
    for i, e in enumerate(ord_index):
        if best_groups[e[0]] == best_groups[e[1]] or edge_scores[args[i]][1] < 0.5:
            continue
        last_index = np.vstack((best_index, e))
        last_loss = grouping_loss(pred_mat, last_index, adj_mat)
        if last_loss < best_loss:
            best_index = last_index
            best_loss = last_loss
            best_groups = node_assignment(last_index, np.ones(len(last_index)), n)

    return best_index, best_loss


def node_assignment_score(edge_index, edge_scores, n):
    """
    Function that finds the graph that produces the lowest
    grouping score by building a score MST and by
    iteratively removing edges that improve the score.

    Args:
        edge_index (np.ndarray) : (E,2) Incidence matrix
        edge_scores (np.ndarray): (E,2) Two-channel edge score
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (E',2) Optimal incidence matrix
    """
    best_index, _ = edge_assignment_score(edge_index, edge_scores, n)
    return node_assignment(best_index, np.ones(len(best_index)), n)