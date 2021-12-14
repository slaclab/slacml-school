from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from sklearn.cluster import DBSCAN
import matplotlib
import time
from sklearn.metrics import log_loss
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


def plot_dataset(dataset,num_image_per_class=10):
    import numpy as np
    num_class = 0
    classes = []
    if hasattr(dataset,'classes'):
        classes=dataset.classes
        num_class=len(classes)
    else: #brute force
        for data,label in dataset:
            if label in classes: continue
            classes.append(label)
        num_class=len(classes)
    
    shape = dataset[0][0].shape
    big_image = np.zeros(shape=[3,shape[1]*num_class,shape[2]*num_image_per_class],dtype=np.float32)
    
    finish_count_per_class=[0]*num_class
    for data,label in dataset:
        if finish_count_per_class[label] >= num_image_per_class: continue
        img_ctr = finish_count_per_class[label]
        big_image[:,shape[1]*label:shape[1]*(label+1),shape[2]*img_ctr:shape[2]*(img_ctr+1)]=data
        finish_count_per_class[label] += 1
        if np.sum(finish_count_per_class) == num_class*num_image_per_class: break
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,8),facecolor='w')
    ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelleft=False,labelbottom=False)
    plt.imshow(np.transpose(big_image,(1,2,0)))
    for c in range(len(classes)):
        plt.text(big_image.shape[1]+shape[1]*0.5,shape[2]*(c+0.6),str(classes[c]),fontsize=16)
    plt.grid(False)
    plt.show()

    
class ImagePadder(object):
    
    def __init__(self,padsize=20,randomize=False):
        self._pads=padsize
        self._randomize=randomize
        
    def __call__(self,img):
        import numpy as np
        img=np.array(img)
        new_shape = list(img.shape)
        new_shape[0] += self._pads
        new_shape[1] += self._pads
        
        new_img = np.zeros(shape=new_shape,dtype=img.dtype)
        pad_vertical   = int(self._pads / 2.)
        pad_horizontal = int(self._pads / 2.)
        if self._randomize:
            pad_vertical   = int(np.random.uniform() * self._pads)
            pad_horizontal = int(np.random.uniform() * self._pads)
        
        new_img[pad_vertical:pad_vertical+img.shape[0],pad_horizontal:pad_horizontal+img.shape[1]] = img
        return new_img
    

# Plot a confusion matrix
def plot_confusion_matrix(labels,prediction,class_names):
    """
    Args:
          prediction ... 1D array of predictions, the length = sample size
          class_names ... 1D array of string label for classification targets, the length = number of categories
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,8),facecolor='w')
    num_labels = len(class_names)
    mat,_,_,im = ax.hist2d(labels,prediction,
                           bins=(num_labels,num_labels),
                           range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(class_names,fontsize=16)
    ax.set_yticklabels(class_names,fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel('True Label',fontsize=20)
    ax.set_ylabel('Prediction',fontsize=20)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, str(mat[i, j]),
                    ha="center", va="center", fontsize=16,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    fig.tight_layout()
    plt.show()


# Compute moving average
def moving_average(a, n=3) :
    
    idx = np.cumsum(np.arange(len(a)),dtype=float)
    idx[n:] = idx[n:] - idx[:-n]
    
    res = np.cumsum(a, dtype=float)
    res[n:] = res[n:] - res[:-n]
    
    return idx[n - 1:] / n, res[n - 1:] / n


# Function to plot softmax score for N-class classification (good for N<10)
def plot_softmax(labels,softmax,class_names):
    import numpy as np
    import matplotlib.pyplot as plt
    num_class   = len(softmax[0])
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
    

from torch.utils.data import Dataset
import os, h5py
class ParticleImage2D(Dataset):

    def __init__(self, data_files, start=0.0, end=1.0, normalize=None):
        
        # Validate file paths
        self._files = [f if f.startswith('/') else os.path.join(os.getcwd(),f) for f in data_files]
        for f in self._files:
            if os.path.isfile(f): continue
            sys.stderr.write('File not found:%s\n' % f)
            raise FileNotFoundError
            
        if start < 0. or start > 1.:
            print('start must take a value between 0.0 and 1.0')
            raise ValueError
        
        if end < 0. or end > 1.:
            print('end must take a value between 0.0 and 1.0')
            raise ValueError
            
        if end <= start:
            print('end must be larger than start')
            raise ValueError

        # Loop over files and scan events
        self._file_handles = [None] * len(self._files)
        self._entry_to_file_index  = []
        self._entry_to_data_index = []
        self._shape = None
        self.classes = []
        self._normalize = normalize
        for file_index, file_name in enumerate(self._files):
            f = h5py.File(file_name,mode='r',swmr=True)
            # data size should be common across keys (0th dim)
            data_size = f['image'].shape[0]
            if not len(f['pdg']) == data_size:
                print(f['image'].shape,len(f['pdg']))
                raise Exception
            self._entry_to_file_index += [file_index] * data_size
            self._entry_to_data_index += range(data_size)

            # if search_pdg is set and pdg exists in data, append unique pdg list
            self.classes += [pdg for pdg in np.unique(f['pdg'])]
            
            f.close()
            
        self.classes = list(np.unique(self.classes))
        self._start  = int(len(self._entry_to_file_index)*start)
        self._length = int(len(self._entry_to_file_index)*end) - self._start

    def __del__(self):
        for i in range(len(self._file_handles)):
            if self._file_handles[i]:
                self._file_handles[i].close()
                self._file_handles[i]=None 


    def __len__(self):
        return self._length
    
    def __getitem__(self,idx):
        file_index  = self._entry_to_file_index[self._start+idx]
        entry_index = self._entry_to_data_index[self._start+idx]
        if self._file_handles[file_index] is None:
            self._file_handles[file_index] = h5py.File(self._files[file_index],mode='r',swmr=True)

        fh = self._file_handles[file_index]

        data = torch.Tensor(fh['image'][entry_index])
        if self._normalize:
            data = (data - self._normalize[0])/self._normalize[1]
        
        # fill the sparse label (if exists)
        label,pdg = None,None
        if 'pdg' in fh:
            pdg = fh['pdg'][entry_index]
            label = self.classes.index(pdg)

        return dict(data=data,label=label,pdg=pdg,index=self._start+idx)


def collate(batch):
    data = torch.cat([sample['data'][None][None] for sample in batch],0)
    label = torch.Tensor([sample['label'] for sample in batch]).long()
    return data,label
