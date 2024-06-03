

def read_czifile(czi_file, squeeze=True):
    
    r""" Reads the data of a simple .czi microscope stack into a numpy array using the lightweight czifile library 

    Parameters
    ----------
    czifile : filepath
        path of the .czi file to read
    squeeze : bool
        specify whether singleton dimensions should be collapsed out

    Returns
    -------
    image_arrays : numpy array
        image stored in the .czi

    """   
    import numpy as np
    from czifile import CziFile
    
    with CziFile(czi_file) as czi:
        image_arrays = czi.asarray()
    if squeeze:
        image_arrays = np.squeeze(image_arrays)
    
    return image_arrays
    
def mkdir(directory):
    
    r"""Recursively creates all directories if they do not exist to make the folder specifed by 'directory'

    Parameters
    ----------
    directory : folderpath
        path of the folder to create
   
    """   

    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return []


def read_pickle(filename):
    r""" read python3 pickled .pkl files

    Parameters
    ----------
    filename : filepath
        absolute path of the file to read
   
    """   
    import pickle
    
    with open(filename, 'rb') as output:
        return pickle.load(output)

def write_pickle(savepicklefile, savedict):

    import pickle

    with open(savepicklefile, 'wb') as handle:
        pickle.dump(savedict, 
                    handle)
        
    return []

def save_segmentation(fname, labels, cmapname='Spectral', n_colors=16):
    
    import skimage.color as skcolor
    import numpy as np 
    import seaborn as sns 
    import skimage.io as skio 
    
    # check whether uint16 or uint32
    if np.max(labels)>65535:
        skio.imsave(fname.replace('.tif', '_labels.tif'), 
                    np.uint32(labels)) # save 32 bit. 
    else:
        skio.imsave(fname.replace('.tif', '_labels.tif'), 
                    np.uint16(labels))
        
    labels_color = np.uint8(255*skcolor.label2rgb(labels, 
                                              colors=sns.color_palette(cmapname, n_colors=n_colors), # spectral is a nicer colorscheme. 
                                              bg_label=0))
    
    skio.imsave(fname.replace('.tif', '_labels-RGB.tif'), 
                labels_color)
    
    return []


