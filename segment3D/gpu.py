# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:37:54 2023

@author: fyz11

helper filters. 
"""

import cupy 
import numpy as np 

def cuda_rescale(im, zoom, order=1, mode='reflect'):
    """
    cuda rescale
    """        
    import cupyx.scipy.ndimage as cupyndimage
    import cupy
    
    out = cupyndimage.zoom(cupy.array(im), zoom=zoom, order=order, mode=mode)
    out = cupy.asnumpy(out)
    
    return out 


def cuda_equalize_adapthist( im, kernel_size=None, clip_limit=0.05,nbins=256):
    
    import cucim.skimage.exposure as cu_skexposure
    
    im_out = cu_skexposure.equalize_adapthist(cupy.array(im), 
                                                kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
    im_out = cupy.asnumpy(im_out)
    
    return im_out


# this also seems good. 
def dask_cuda_rescale(img, zoom, order=1, mode='reflect', chunksize=(512,512,512)):    
    import dask.array as da

    im_chunk = da.from_array(img, chunks=chunksize) # make into chunk -> we can then map operation?  
    g = im_chunk.map_blocks(cuda_rescale, zoom=zoom, order=order, mode=mode)
    result = g.compute()
    
    return result
    

def dask_cuda_bg(img, bg_ds=8, bg_sigma=5, chunksize=(512,512,512)):    
    """ Estimates and removes an estimated background based on filtering
    
    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    bg_ds : TYPE, optional
        DESCRIPTION. The default is 8.
    bg_sigma : TYPE, optional
        DESCRIPTION. The default is 5.
    chunksize : TYPE, optional
        DESCRIPTION. The default is (512,512,512).

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    import dask.array as da
    import scipy.ndimage as ndimage 
    # import cucim.skimage.transform as cu_transform # seems to not need this and this seems experimental / temperamental. 

    im_chunk = da.from_array(img, chunks=chunksize) # make into chunk -> we can then map operation?  
    g = im_chunk.map_blocks(cuda_rescale, zoom=[1./bg_ds]*len(img.shape), order=1, mode='reflect', dtype=np.float32)
    # now do the gaussian filter
    g = g.map_overlap(ndimage.gaussian_filter, sigma=bg_sigma, depth=2*bg_sigma, boundary='reflect', dtype=np.float32).compute()  # we need to compute in order to get the scaling.
    
    # we might be able to do the proper scaling in this space.... 
    im_chunk = da.from_array(g, chunks=chunksize)
    g = im_chunk.map_blocks(cuda_rescale, zoom=np.hstack(img.shape)/(np.hstack(im_chunk.shape)), order=1, mode='reflect', dtype=np.float32)
    
    result = g.compute()
    result = num_bg_correct(img,result, eps=1e-8)
    return result


def cuda_smooth_vol( im, ds=4, sigma=5):
    
    import cupyx.scipy.ndimage as ndimage
    import cucim.skimage.transform as cu_transform
    import numpy as np 
    
    out = cu_transform.resize(cupy.array(im), np.array(im.shape)//ds, preserve_range=True)
    out = ndimage.gaussian_filter(out, sigma=sigma)
    out = cu_transform.resize(out, np.array(im.shape), preserve_range=True) 
    
    return cupy.asnumpy(out)


def cuda_normalize(x, pmin=2, pmax=99.8, axis=None,  clip=False, eps=1e-20, cast_numpy=False):
    
    mi, ma = cupy.percentile(cupy.array(x),[pmin,pmax],axis=axis,keepdims=True)
    out = (cupy.array(x) - mi) / ( ma - mi + eps )
    
    if clip:
        out = cupy.clip(out, 0, 1)
        
    if cast_numpy:
        return cupy.asnumpy(out)
    else:
        return out
        
    
# import dask.array as da
from numba import jit
@jit(nopython=True, parallel=True)
def num_bg_correct(im, bg, eps=0.1):
    return np.mean(bg)/(bg + eps) * im

    
@jit(nopython=True, nogil=True, parallel=True)
def num_normalize(a, pmin=2, pmax=99.8, clip=True, eps=1e-20):
    
    # mi = np.percentile(a, pmin)
    mi, ma = np.percentile(a, [pmin, pmax])
    a = (a-mi)/(ma-mi+eps)
    if clip:
        a = np.clip(a,0,1)
    return a.astype(np.float32)
    

# @jit(nopython=True, nogil=True, parallel=True)
# def fill_array(im, thresh=0, fill_vals=0, method='constant'):
    
#     import numpy as np 
#     out = im.copy()
    
#     if method == 'constant':
#         out[im<=thresh] = fill_vals
#     if method == 'median':
#         out[im<=thresh] = np.nanmedian(im[im>thresh])
#     if method == 'mean':
#         out[im<=thresh] = np.nanmean(im[im>thresh])
        
#     return out

def bg_normalize(im, bg_ds=8, bg_sigma=5):
    
    # single channel .
    bg = cuda_smooth_vol(im, ds=bg_ds, sigma=bg_sigma)
    
    # get the normalized. 
    corrected = num_bg_correct(im, bg)
    
    return corrected.astype(np.float32) 


# reproducing here for convenience 
def _smooth_vol(vol_binary, ds=4, sigma=5):
    
    from scipy.ndimage import gaussian_filter
    import skimage.transform as sktform
    import numpy as np 
    
    small = sktform.resize(vol_binary, np.array(vol_binary.shape)//ds, preserve_range=True)
    small = gaussian_filter(small, sigma=sigma)
    
    return sktform.resize(small, np.array(vol_binary.shape), preserve_range=True)


def bg_normalize_cpu(im, bg_ds=4, bg_sigma=5):
    
    # single channel .
    bg = _smooth_vol(im, ds=bg_ds, sigma=bg_sigma)
    
    # get the normalized. 
    corrected = num_bg_correct(im, bg)
    
    return corrected.astype(np.float32) 
    
    
    
    
    
    
    
