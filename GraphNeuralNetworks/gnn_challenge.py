import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset

def plot_scatter3d(voxels, labels={}, edge_index=None, clust_labels=None, markersize=5, colorscale=None):
    
    import copy
    import plotly.graph_objects as go # Plotly graph objects
    from plotly.offline import iplot  # Interactive plots
    import plotly.express as px # For discrete color scale
    
    if colorscale is None:
        colorscale = px.colors.qualitative.Dark24

    # Initialize a graph with no labels
    blank = go.Scatter3d(x = voxels[:,0],
                         y = voxels[:,1],
                         z = voxels[:,2],
                         mode = 'markers', 
                         marker = dict(
                             size = 2, 
                             colorscale = colorscale
                         )
                        )
    
    # Loop over the set of labels, make a graph for each
    graphs = []
    for key, label in labels.items():
        graph = copy.copy(blank)
        graph['name'] = key
        graph['marker']['color'] = label
        graph['hovertext'] = [f'{l:0.0f}' for l in label]
        graphs.append(graph)
        
    if not len(graphs):
        graphs.append(blank)
        
    # Draw edges if requested
    if edge_index is not None:
        import scipy as sp
        edge_vertices = []
        clust_labels = np.unique(clust_labels, return_inverse=True)[1]
        for i, j in edge_index:
            vi, vj = voxels[clust_labels==i], voxels[clust_labels==j]
            d12 = sp.spatial.distance.cdist(vi, vj, 'euclidean')
            i1, i2 = np.unravel_index(np.argmin(d12), d12.shape)
            edge_vertices.append([vi[i1], vj[i2], [None, None, None]])

        edge_vertices = np.concatenate(edge_vertices)

        graphs.append(go.Scatter3d(x = edge_vertices[:,0], y = edge_vertices[:,1], z = edge_vertices[:,2],
                                   mode = 'lines',
                                   name = 'Predicted edges',
                                   line = dict(
                                        width = 2,
                                        color = 'Blue'
                                    ),
                                    hoverinfo = 'none'))

    iplot(graphs)

class ShowerVoxels(Dataset):
    """
    class: an interface to access shower fragment voxels. This Dataset is designed to produce a shower voxels
    """
    def __init__(self, file_path):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._file_handle = None
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['voxels'])
        
    def __del__(self):
        
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __len__(self):
        
        return self._entries

    def __getitem__(self, idx):
            
        # Get the subset of voxels that correspond to the requested entry
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)
            
        voxel_info = self._file_handle['voxels'][idx].reshape(-1,6)
        clusts = [np.where(voxel_info[:,3]==c)[0] for c in np.unique(voxel_info[:,3])]
        voxels = [torch.tensor(voxel_info[c,:3], dtype=torch.float32) for c in clusts]
        frag_ids = np.array([voxel_info[c[0],3] for c in clusts], dtype=np.int64)
        shower_ids = np.array([voxel_info[c[0],4] for c in clusts], dtype=np.int64)
        primary_ids = np.array([voxel_info[c[0],5] for c in clusts], dtype=np.int64)
            
        return dict(
            voxels = voxels,
            frag_ids = frag_ids,
            shower_ids = shower_ids,
            primary_ids = primary_ids
        )


def collate_voxels(batch):
    
    import itertools
    
    return dict(
        voxels = list(itertools.chain.from_iterable([b['voxels'] for b in batch])),
        frag_ids = np.concatenate([b['frag_ids'] for b in batch]),
        shower_ids = np.concatenate([b['shower_ids'] for b in batch]),
        primary_ids = np.concatenate([b['primary_ids'] for b in batch]),
        batch = np.concatenate([i*np.ones(len(b['frag_ids']), dtype=np.int64) for i, b in enumerate(batch)])
        )

class ShowerFeatures(GraphDataset):
    """
    class: an interface for shower fragment data files. This Dataset is designed to produce a batch of
           of node and edge feature data.
    """
    def __init__(self, file_path):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._file_handle = None
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['node_features'])

    def __del__(self):
        
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            
    def __len__(self):
        
        return self._entries

    def __getitem__(self, idx):
        
        # Get the subset of node and edge features that correspond to the requested event ID
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)
            
        node_info = torch.tensor(self._file_handle['node_features'][idx].reshape(-1, 19), dtype=torch.float32)
        node_features, group_ids, node_labels = node_info[:,:-3], node_info[:,-2].long(), node_info[:,-1].long()
        
        edge_info = torch.tensor(self._file_handle['edge_features'][idx].reshape(-1, 22), dtype=torch.float32)
        edge_features, edge_index, edge_labels = edge_info[:,:-3], edge_info[:,-3:-1].long().t(), edge_info[:,-1].long()
        
        return GraphData(x = node_features,
                         edge_index = edge_index,
                         edge_attr = edge_features,
                         y = node_labels,
                         edge_label = edge_labels,
                         index = idx)