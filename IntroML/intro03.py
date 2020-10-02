import numpy as np
import torch

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
    plt.show()

# Plot a confusion matrix
def plot_confusion_matrix(label,prediction,class_names):
    """
    Args: label ... 1D array of true label value, the length = sample size
          prediction ... 1D array of predictions, the length = sample size
          class_names ... 1D array of string label for classification targets, the length = number of categories
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,8),facecolor='w')
    num_labels = len(class_names)
    max_value  = np.max([np.max(np.unique(label)),np.max(np.unique(label))])
    assert max_value < num_labels
    mat,_,_,im = ax.hist2d(label,prediction,
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
    ax.set_xlabel('Prediction',fontsize=20)
    ax.set_ylabel('True Label',fontsize=20)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, str(mat[i, j]),
                    ha="center", va="center", fontsize=16,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    fig.tight_layout()
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

# Compute moving average
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Function to plot softmax score for N-class classification (good for N<10)
def plot_softmax(label,softmax):
    import numpy as np
    import matplotlib.pyplot as plt
    num_class   = len(softmax[0])
    unit_angle  = 2*np.pi/num_class
    xs = np.array([ np.sin(unit_angle*i) for i in range(num_class+1)])
    ys = np.array([ np.cos(unit_angle*i) for i in range(num_class+1)])
    fig,axis=plt.subplots(figsize=(10,10),facecolor='w')
    plt.plot(xs,ys)#,linestyle='',marker='o',markersize=20)
    for d in range(num_class):
        plt.text(xs[d]*1.1-0.04,ys[d]*1.1-0.04,str(d),fontsize=24)
    plt.xlim(-1.3,1.3)
    plt.ylim(-1.3,1.3)
    
    xs=xs[0:10]
    ys=ys[0:10]
    for d in range(num_class):
        idx=np.where(label==d)
        scores=softmax[idx]
        xpos=[np.sum(xs * s) for s in scores]
        ypos=[np.sum(ys * s) for s in scores]
        plt.plot(xpos,ypos,linestyle='',marker='o',markersize=10,alpha=0.5)
    axis.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelleft=False,labelbottom=False)
    plt.show()

