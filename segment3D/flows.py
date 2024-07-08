# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 03:04:09 2023

@author: fyz11

Module for computing and implementing various flow definitions. 

"""

import numpy as np 


"""
Flow 1 - euclidean distance transform. 
"""
# def distance_transform_labels(labels, bg_label=0):
    
#     import numpy as np 
#     from scipy.ndimage import distance_transform_edt
#     # import skfmm
    
#     dtform = np.zeros(labels.shape)
#     dtform_flow = np.zeros((2,)+labels.shape) # keep these separate!. 
    
#     uniq_labels = np.setdiff1d(np.unique(labels), bg_label)
    
#     for lab in uniq_labels:
        
#         mask = labels == lab
#         # dist_mask = skfmm.distance(mask>0)
#         # dist_mask = distance_transform_edt(mask>0)
#         dist_mask = sdf_distance_transform(mask>0)
        
#         # compute the gradient!. 
#         grad_dist_mask = np.array(np.gradient(dist_mask))
#         grad_dist_mask = grad_dist_mask / (np.linalg.norm(grad_dist_mask, axis=0)[None,...] + 1e-20)
        
#         dtform_flow[:,mask>0] = grad_dist_mask[:,mask>0].copy()
        
#         dtform[mask>0] = dist_mask[mask>0]
        
#     return dtform_flow, dtform 


# =============================================================================
#  Use the euclidean distance transform given in the edt library which is faster.
# =============================================================================
def distance_transform_labels_fast(labels, sdf=False, n_threads=16, black_border=False):
    """ compute euclidean distance transform for each uniquely labelled cell in a 2D/3D binary image 
    
    """
    import numpy as np 
    import edt # external fast distance transform from seung lab. https://github.com/seung-lab/euclidean-distance-transform-3d # can also compute sdf?
    
    if not sdf:
        
        if len(labels.shape) == 2:
            dtform = edt.edt(labels, 
                             black_border=black_border, 
                             order='C', 
                             parallel=n_threads).astype(np.float32)    
        else:
            # does this not work for a 3D label set?
            dtform = np.array([edt.edt(ss, black_border=black_border, 
                                       order='C', 
                                       parallel=n_threads).astype(np.float32) for ss in labels])
    else:
        if len(labels.shape) == 2:
            dtform = edt.sdf(labels, black_border=black_border,
                             order='C', 
                             parallel=n_threads).astype(np.float32)
        else:
            dtform = np.array([edt.sdf(ss, black_border=black_border, 
                                       order='C', 
                                       parallel=n_threads).astype(np.float32) for ss in labels])
        
    return dtform 



# =============================================================================
#   Use the poisson heat distance transform 
# =============================================================================

# construct the laplacian grid for a 2D image. (might be faster to exploit kron! )
def _laplacian_matrix(n, m, mask=None): # this is the only working solution! 
    
    import scipy.sparse
    from scipy.sparse.linalg import spsolve
    
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
    
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    if mask is not None:
        y_range = mask.shape[0]
        x_range = mask.shape[1]
        # find the masked i.e. zeros
        zeros = np.argwhere(mask==0) # in (y,x)
        k = zeros[:,1] + zeros[:,0] * x_range
        mat_A[k,k] = 1
        mat_A[k, k + 1] = 0
        mat_A[k, k - 1] = 0
        mat_A[k, k + x_range] = 0
        mat_A[k, k - x_range] = 0
        
        mat_A = mat_A.tocsc()
    else:
        mat_A = mat_A.tocsc()
    
    return mat_A

def poisson_dist_tform(binary, pt=None):
    """
    Computation for a single binary image. 
    """
    import scipy.sparse
    from scipy.sparse.linalg import spsolve
    import numpy as np 
    
    mask = np.pad(binary, [1,1], mode='constant', constant_values=1) # pad with ones.  # ones need for connectivity....
    
    mat_A = _laplacian_matrix(mask.shape[0], mask.shape[1], mask=mask) # this is correct!

    if pt is not None:
        mask = np.zeros_like(mask)
        mask[pt[:,0], pt[:,1]] = 1
        
        mask_flat = mask.flatten()    
        mat_b = np.zeros(len(mask_flat))
        mat_b[mask_flat == 1] = 1
        
    else:
        # now we need to zero the padding
        mask[0,:]=0
        mask[:,-1]=0
        mask[-1,:]=0
        mask[:,0]=0
        
        # solve within the mask!    
        mask_flat = mask.flatten()    
        # inside the mask:
        # \Delta f = div v = \Delta g       
        mat_b = np.ones(len(mask_flat)) # does this matter -> the amount... 
        mat_b[mask_flat == 0] = 0
        
    x = spsolve(mat_A, mat_b, permc_spec='MMD_AT_PLUS_A') 
    x = x.reshape(mask.shape)
    x = x[1:-1,1:-1].copy()
    x = x - x.min() # solution is only positive!.
    
    return x 
    
def poisson_dist_tform_flow(coords, centroid, shape, power_dist=None, norm_dist=False):
    """
    Solve diffusion for points. 
    """
    import numpy as np 
    import scipy.ndimage as ndimage
    
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:,0], coords[:,1]] = 1
    
    poisson = poisson_dist_tform(mask>0, pt=centroid[None,:]) # compute the point source distance transform 
    poisson = ndimage.gaussian_filter(poisson, sigma=1)

    # if log_dist:
        # poisson = np.log(poisson+1.)
    if power_dist is not None:
        # why are we getting nan? wiht power boost? 
        # this is to boost beyond the numerical limits. 
        
        poisson = np.clip(poisson, 0, 1)
        poisson = poisson**power_dist # valid for 0-1
        
    poisson_flow = np.array(np.gradient(poisson))

    dy = poisson_flow[0, coords[:,0], coords[:,1]].copy()
    dx = poisson_flow[1, coords[:,0], coords[:,1]].copy()
    
    return poisson[coords[:,0], coords[:,1]], np.stack((dy,dx))


def poisson_dist_tform_flow_sets(coords, pts, shape, power_dist=None):
    """
    Solve diffusion for points. 
    """
    import numpy as np 
    import scipy.ndimage as ndimage
    
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:,0], coords[:,1]] = 1
    
    poisson = poisson_dist_tform(mask>0, pt=pts) # compute the point source distance transform 
    poisson = ndimage.gaussian_filter(poisson, sigma=1) # i smooth? O.O

    # if log_dist:
        # poisson = np.log(poisson+1.)
    if power_dist is not None:
        # why are we getting nan? wiht power boost? 
        # this is to boost beyond the numerical limits. 
        poisson = (poisson-poisson.min())
        # poisson = np.clip(poisson, 0, 1)
        poisson = poisson**power_dist # valid for 0-1
        
    poisson_flow = np.array(np.gradient(poisson))
    poisson_flow[:, pts[:,0], pts[:,1]] = 0 #enforce the limit set. 
    
    dy = poisson_flow[0, coords[:,0], coords[:,1]].copy()
    dx = poisson_flow[1, coords[:,0], coords[:,1]].copy()
    
    return poisson[coords[:,0], coords[:,1]], np.stack((dy,dx))


# =============================================================================
#   Defining the fmm geodesic flow
# =============================================================================
# def fmm_point_source2D(binary, pt):
#     """
#     This is a much better solver of the diffusion
#     """
    
#     import skfmm
#     import numpy as np
#     import skimage.morphology as skmorph
#     import scipy.ndimage as ndimage 
    
#     # create a masked array
#     # mask = np.logical_not(skmorph.binary_dilation(binary, skmorph.disk(1)))
#     mask = np.logical_not(binary).copy()
#     # mask2 = np.logical_not(skmorph.binary_dilation(binary, skmorph.disk(1)))
#     mask2 = mask.copy()
#     m = np.ones_like(binary)
#     m[pt[0],pt[1]] = 0
#     m = np.ma.masked_array(m, mask2)

#     dist_image = skfmm.distance(m)  # this is the answer!!!     
#     dist_image = dist_image.max()-dist_image # invert the image!. # many ways to do this. 
    
#     # fix up the boundary differentiation. 
#     dist_outer = (skfmm.distance(mask)*-1) # can we make it smoother? # best
#     dist_image[mask>0] = dist_outer[mask>0] # this seems to work! (it gives a little normal push. ) 
#     # dist_image = dist_image/dist_image.max() # retain this. 
    
#     dist_gradient = np.array(np.gradient(dist_image))
#     # dist_gradient = dist_gradient/(np.linalg.norm(dist_gradient, axis=0)[None,...]+1e-20) # normalise. 
#     dist_gradient[:,mask>0] = 0
#     dist_image[mask>0]=0
    
#     return dist_image, dist_gradient

def fmm_point_source2D(coords, centroid, shape):
    """
    This is a much better solver of the diffusion
    """
    
    import skfmm
    import numpy as np
    import skimage.morphology as skmorph
    import scipy.ndimage as ndimage 
    
    # construct the binary 
    binary = np.zeros(shape, dtype=bool)
    binary[coords[:,0], coords[:,1]] = 1
    
    # construct the masked out region!. 
    mask = np.logical_not(binary).copy()
    # mask2 = np.logical_not(skmorph.binary_dilation(binary, skmorph.disk(1)))
    mask2 = mask.copy()
    m = np.ones_like(binary)
    m[centroid[0],centroid[1]] = 0 # define the point source!. 
    m = np.ma.masked_array(m, mask2)

    dist_image = skfmm.distance(m)  # this is the answer!!!     
    dist_image = dist_image.max()-dist_image # invert the image!. # many ways to do this. 
    
    # fix up the boundary differentiation. 
    dist_outer = (skfmm.distance(mask)*-1) # can we make it smoother? # best
    dist_image[mask>0] = dist_outer[mask>0] # this seems to work! (it gives a little normal push. ) 
    # dist_image = dist_image/dist_image.max() # retain this. 
    
    dist_gradient = np.array(np.gradient(dist_image))
    
    dy = dist_gradient[0, coords[:,0], coords[:,1]].copy()
    dx = dist_gradient[1, coords[:,0], coords[:,1]].copy()
    
    return dist_image[coords[:,0], coords[:,1]], np.stack((dy,dx))
    

def _largest_component_vol(vol_binary, connectivity=1):
    r""" Given a binary segmentation, return only the largest connected component of the given connectivity

    Parameters
    ----------
    vol : array
        input binary image 
    connectivity : 1 or 2
        if 1, the local 4-neighbors for 2D or 6-neighbors for 3D. 
        if 2, the local 8-neighbors for 2D or 26-neighbors for 3D. 
   
    Returns
    ------- 
    vol_binary : array
        output binary image same size as input retaining only the largest connected component

    """
    from skimage.measure import label, regionprops
    import numpy as np 
    
    vol_binary_labelled = label(vol_binary, connectivity=connectivity)
    # largest component.
    vol_binary_props = regionprops(vol_binary_labelled)
    vol_binary_vols = [re.area for re in vol_binary_props]
    vol_binary = vol_binary_labelled == (np.unique(vol_binary_labelled)[1:][np.argmax(vol_binary_vols)])
    
    return vol_binary



def fmm_point_source2D_sets(coords, source_pts, shape):
    """
    This is a much better solver of the diffusion
    """
    
    import skfmm
    import numpy as np
    import skimage.morphology as skmorph
    import scipy.ndimage as ndimage 
    
    # construct the binary 
    binary = np.zeros(shape, dtype=bool)
    binary[coords[:,0], coords[:,1]] = 1
    
    # construct the masked out region!. 
    mask = np.logical_not(binary).copy()
    # mask2 = np.logical_not(skmorph.binary_dilation(binary, skmorph.disk(1)))
    mask2 = mask.copy()
    m = np.ones_like(binary)
    m[source_pts[:,0],source_pts[:,1]] = 0 # define the set source!. # this is not just a single point. 
    m = np.ma.masked_array(m, mask2)

    dist_image = skfmm.distance(m)  # this is the answer!!!     
    dist_image = dist_image.max()-dist_image # invert the image!. # many ways to do this. 
    # max_image = dist_image.max()
    
    # fix up the boundary differentiation. 
    dist_outer = (skfmm.distance(mask)*-1) # can we make it smoother? # best
    dist_image[mask>0] = dist_outer[mask>0] # this seems to work! (it gives a little normal push. ) 
    # dist_image = dist_image/dist_image.max() # retain this. 
    
    
    """
    normalize the distance !
    """
    # dist_image = dist_image/np.abs(max_image)
    
    dist_gradient = np.array(np.gradient(dist_image))
    dist_gradient[:,source_pts[:,0], source_pts[:,1]] = 0 # enforce to be zero !
    
    dy = dist_gradient[0, coords[:,0], coords[:,1]].copy()
    dx = dist_gradient[1, coords[:,0], coords[:,1]].copy()
    
    return dist_image[coords[:,0], coords[:,1]], np.stack((dy,dx))



# =============================================================================
#   Replicating cellpose flow
# =============================================================================
from numba import njit, float32, int32, vectorize
@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T,y,x,ymed,xmed,Lx, niter):
    """ run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)
    
    taken from  cellpose source code.
    
    Parameters
    --------------
    T: float64, array
        _ x Lx array that diffusion is run in
    y: int32, array
        pixels in y inside mask
    x: int32, array
        pixels in x inside mask
    ymed: int32
        center of mask in y
    xmed: int32
        center of mask in x
    Lx: int32
        size of x-dimension of masks
    niter: int32
        number of iterations to run diffusion
    Returns
    ---------------
    T: float64, array
        amount of diffused particles at each pixel
    """

    for t in range(niter):
        T[ymed*Lx + xmed] += 1
        T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                                            T[y*Lx + x-1]     + T[y*Lx + x+1] +
                                            T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
                                            T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
    return T


def cellpose_diffuse2D(coords, centroid, shape, niter):
    """
    Solve diffusion for points. 
    """
    import numpy as np 
    
    ly, lx = shape
    
    x = coords[:,1].copy() # these are the diffusion points to solve for. 
    y = coords[:,0].copy()

    T = np.zeros((ly+2)*(lx+2), np.float64)
    T = _extend_centers(T,
                        y=y.astype(np.int32),
                        x=x.astype(np.int32),
                        ymed=centroid[0].astype(np.int32),
                        xmed=centroid[1].astype(np.int32),
                        Lx=int32(lx), 
                        niter=int32(niter))
    T[(y+1)*lx + x+1] = np.log(1.+T[(y+1)*lx + x+1])
    dy = T[(y+1)*lx + x] - T[(y-1)*lx + x]
    dx = T[y*lx + x+1] - T[y*lx + x-1]
    
    return T[(y+1)*lx + x+1], np.stack((dy,dx))


# =============================================================================
#   Generic wrapper functions.  
# =============================================================================

def distance_tform_labels2D(labelled, dtform_fnc, clip=False): 
    """
    this function wraps a distance transform defined for a single binary and generalize to multi-labels. 
    """
    import skimage.measure as skmeasure
    import numpy as np 
    
    distance = np.zeros(labelled.shape, dtype=np.float32)
    # distance = np.zeros(labelled.shape)
    binary = labelled>0
    
    labelled_regions = skmeasure.regionprops(labelled)
    
    for reg in labelled_regions:
        
        patch = reg.image
        box = reg.bbox # (min_row, min_col, max_row, max_col)
        y1, x1, y2, x2 = box
        coords = reg.coords
        
        dist = dtform_fnc(patch) 
        
        if clip:
            dist = dist / dist.max()
            dist = np.clip(dist, 0, 1)
        else:
            dist = np.clip(dist, 0, np.inf)
        
        # copy it into the larger array. 
        distance[coords[:,0], coords[:,1]] = dist[coords[:,0]-y1, coords[:,1]-x1].copy()
        
    return distance


def distance_centroid_tform_flow_labels2D(labelled, 
                                          dtform_method='cellpose', 
                                          guide_img=None, 
                                          fixed_point_percentile=0.9, 
                                          iter_factor=5,
                                          smooth_skel_sigma=3,
                                          power_dist=None):
    
    """
    This function wraps a distance transform defined for a single binary with point source and generalize to multi-labels. 
    """
    # is there a better way to smooth this? 
    import skimage.measure as skmeasure
    from skimage.morphology import skeletonize
    import skimage.segmentation as sksegmentation 
    import skimage.morphology as skmorph
    import numpy as np 
    import scipy.ndimage as ndimage
    # import skimage.filters as skfilters

    distance = np.zeros(labelled.shape)
    flow = np.zeros((2,)+labelled.shape)
    # binary = labelled>0 # to tell us which is not background 
    
    labelled_regions = skmeasure.regionprops(labelled)
    
    for reg in labelled_regions:
        patch = reg.image
        box = reg.bbox # (min_row, min_col, max_row, max_col) # same behavior
        y1, x1, y2, x2 = box
        start = np.hstack([y1,x1])
        coords = reg.coords 
        
        if len(coords)>8: # minimum size. 
            # determine the median 
            centroid = np.nanmedian(coords, axis=0)
            
            """
            stability for gauss-siedel methods. 
            """
            inner = sksegmentation.find_boundaries(patch*1, mode='inner')
            inner = np.logical_and(patch, np.logical_not(inner))
            inner_coords = np.argwhere(inner>0)
            
            if dtform_method == 'fmm_skel' or dtform_method == 'cellpose_skel':
    
                # we compute the skeleton. 
                # print('computing skel')
                # # replicate the local binary mask 
                # patch_mask = np.zeros((y2-y1+1, x2-x1+1), dtype=bool)
                # patch_mask[coords[:,0]-start[0], coords[:,1]-start[1]] = 1
                # patch_mask = skmorph.binary_erosion(inner, skmorph.square(1)).copy() # compute on the inner!. 
                patch_mask = inner.copy()
                # smooth and rebinarize... 
                
                if smooth_skel_sigma > 0: 
                    patch_mask = ndimage.gaussian_filter(patch_mask*255., sigma=smooth_skel_sigma) # remove curvature... 
                    # print(patch_mask.min(), patch_mask.max())
                    patch_mask = patch_mask/(patch_mask.max()+1e-12)> 0.5
                    # patch_mask = skmorph.binary_erosion(patch_mask, skmorph.disk(1)).copy() # compute on the inner!. 
                    skel = skmorph.skeletonize(patch_mask)
                    skel_base = skel.copy()
                    # smooth and reskel
                    skel = ndimage.gaussian_filter(skel*255., sigma=smooth_skel_sigma)
                    skel = skel/(skel.max()+1e-12) 
                    # print('computing skel:, ', np.mean(skel))
                    skel = skel > np.mean(skel)
                    # skel = skel > np.median(skel)
                    # skel = skel > skfilters.threshold_otsu(skel)
                    try:
                        skel = _largest_component_vol(skel)
                        skel = skmorph.skeletonize(skel) # this is 1 line absolutely constrained. 
                    except:
                        skel = skel_base.copy()
                    skel = skmorph.binary_dilation(skel, skmorph.disk(1))
                else:
                    skel = skmorph.skeletonize(patch_mask)
                    skel = skmorph.binary_dilation(skel, skmorph.disk(1))
                    
            if len(inner_coords)>0:
                inner_coords = (inner_coords+start[None,:]).copy()
            else:
                inner_coords = coords.copy()
            
            if guide_img is not None:
                # # use this to find an internal fixed point. 
                coords_guide = guide_img[inner_coords[:,0],inner_coords[:,1]].copy()
                coords_valid = inner_coords[coords_guide>=np.percentile(coords_guide, fixed_point_percentile*100)]
                # coords_valid = np.argwhere(skeletonize(patch>0)>0)+start[None,:]
                centroid = coords_valid[np.argmin(np.linalg.norm(coords_valid-centroid[None,:], axis=1))]
            else:
                centroid = inner_coords[np.argmin(np.linalg.norm(inner_coords-centroid[None,:], axis=1))]
                
                
            if dtform_method == 'cellpose':
                niter = iter_factor*np.int32(np.ptp(coords[:,1]) + np.ptp(coords[:,0])) # there should be a better estimate of this. 
                
                dist_, flow_ = cellpose_diffuse2D(coords-start[None,:]+1, # remove the global offset. 
                                               centroid-start+1,
                                                shape=np.hstack(patch.shape)+1, 
                                                niter=niter)
            
            if dtform_method == 'cellpose_improve':    
                dist_, flow_ = poisson_dist_tform_flow(coords-start[None,:]+1, # remove the global offset. 
                                                       centroid-start+1,
                                                       shape=np.hstack(patch.shape)+1,
                                                       power_dist=power_dist)
            
            if dtform_method == 'fmm':  
                # geodesic fast marching method
                dist_, flow_ = fmm_point_source2D(coords-start[None,:]+1, # remove the global offset. 
                                                       centroid-start+1,
                                                       shape=np.hstack(patch.shape)+1)
                
                
            if dtform_method == 'fmm_skel':
                # print(np.sum(skel>0))
                if np.sum(skel>0) >0:    
                    dist_, flow_ = fmm_point_source2D_sets(coords-start[None,:]+1, # remove the global offset. 
                                                           np.argwhere(skel>0)+1,
                                                           shape=np.hstack(patch.shape)+1)
                else:
                    dist_ = np.zeros(len(coords))
                    flow_ = np.zeros((2,len(coords)))
            if dtform_method == 'cellpose_skel':
                
                if np.sum(skel>0) >0:    
                    # skel = skmorph.binary_dilation(skel,skmorph.disk(1))  # expand. 
                    dist_, flow_ = poisson_dist_tform_flow_sets(coords-start[None,:]+1, # remove the global offset. 
                                                                np.argwhere(skel>0)+1,
                                                                shape=np.hstack(patch.shape)+1,
                                                                power_dist=power_dist)
                else:
                    dist_ = np.zeros(len(coords))
                    flow_ = np.zeros((2,len(coords)))
            # diffuse_out = out.reshape(ly+2,lx+2, order='')
            distance[coords[:,0],coords[:,1]] = dist_.copy()
            flow[:,coords[:,0],coords[:,1]] = flow_.copy()
        
    # normalise
    flow /= (1e-20 + (flow**2).sum(axis=0)**0.5)
        
    return np.concatenate([distance[None,...], flow], axis=0) # concatenate this!.  # then we compute both!. 


# =============================================================================
# Parallel scripts
# =============================================================================

def _distance_centroid_tform_flow_labels2D_chunk(labelled_array, 
                                                 dtform_method='cellpose', 
                                                 guide_img=None,
                                                 fixed_point_percentile=0.9,
                                                 iter_factor=5,
                                                 smooth_skel_sigma=3,
                                                 power_dist=None):

     import numpy as np

     if guide_img is not None:
         dtform_flow = np.array([distance_centroid_tform_flow_labels2D(labelled_array[zz], 
                                                                       dtform_method = dtform_method,
                                                                       guide_img=guide_img[zz], 
                                                                       fixed_point_percentile=fixed_point_percentile,
                                                                       iter_factor=iter_factor,
                                                                       smooth_skel_sigma=smooth_skel_sigma,
                                                                       power_dist=power_dist) for zz in np.arange(len(labelled_array))])
     else:
         dtform_flow = np.array([distance_centroid_tform_flow_labels2D(labelled_array[zz], 
                                                                       dtform_method = dtform_method,
                                                                       guide_img=None, 
                                                                       fixed_point_percentile=fixed_point_percentile,
                                                                       iter_factor=iter_factor,
                                                                       smooth_skel_sigma=smooth_skel_sigma,
                                                                       power_dist=power_dist) for zz in np.arange(len(labelled_array))])
     return dtform_flow
 
    
def distance_centroid_tform_flow_labels2D_parallel(labelled_array, 
                                                   dtform_method='cellpose',
                                                    guide_image=None,
                                                    fixed_point_percentile=0.9,
                                                    iter_factor=5, 
                                                    n_processes=4,
                                                    smooth_skel_sigma=3,
                                                    power_dist=None):
    # import multiprocessing as mp
    import multiprocess as mp
    import numpy as np 
    
    n_proc = n_processes
    chunksize = len(labelled_array)//n_proc
    
    # chunk the data. 
    chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None
        chunks.append(labelled_array[slice(chunkstart, chunkend)])
 
    assert sum(map(len, chunks)) == len(labelled_array)
    print('starting parallel')
         
    # parallel computing 
    with mp.Pool(processes=n_proc) as pool:
        # starts the sub-processes without blocking
        # pass the chunk to each worker process
        # proc_results = [pool.apply_async(centroid_geodesic_distance_transform2D_chunk,
        #                                     args=(chunk,))
        #                                   for chunk in chunks]
        
        proc_results = [pool.apply_async(_distance_centroid_tform_flow_labels2D_chunk,
                                            args=(chunk, dtform_method, guide_image,fixed_point_percentile, iter_factor,smooth_skel_sigma, power_dist))
                                          for chunk in chunks]
        # if dtform_method == 'cellpose_improve':
        #     proc_results = [pool.apply_async(_distance_centroid_tform_flow_labels2D_chunk,
        #                                         args=(chunk, dtform_method, guide_image,fixed_point_percentile, smooth_skel_sigma, power_dist))
        #                                       for chunk in chunks]
        # else:
        #     proc_results = [pool.apply_async(_distance_centroid_tform_flow_labels2D_chunk,
        #                                        args=(chunk, dtform_method, guide_image,fixed_point_percentile, iter_factor))
        #                                      for chunk in chunks]
         
        # blocks until all results are fetched
        res = np.vstack([r.get() for r in proc_results])
         
    return res


def distance_centroid_tform_flow_labels2D_dask(labelled_array, 
                                                   dtform_method='cellpose_improve',
                                                    guide_image=None,
                                                    fixed_point_percentile=0.9,
                                                    iter_factor=5, 
                                                    n_processes=4,
                                                    smooth_skel_sigma=3,
                                                    power_dist=None):
    # import multiprocessing as mp
    import numpy as np 
    import dask
    from dask.distributed import Client
    
    n_proc = n_processes
    client = Client(n_workers=n_proc, threads_per_worker=1)
    # client = Client(n_workers=n_proc)
    # client = Client(n_workers=n_proc, threads_per_worker=1) # enforce single thread, this reduces garbage collection
    # client = Client(processes=False) # since we are just doing numpy
    
    lazy_flow = dask.delayed(distance_centroid_tform_flow_labels2D)
    
    
    res = []
    for sli in np.arange(len(labelled_array)):
        if dtform_method == 'cellpose_improve':
            if guide_image is not None:
                res.append(lazy_flow(labelled_array[sli], 
                                         dtform_method=dtform_method, 
                                         guide_img=guide_image[sli], 
                                         fixed_point_percentile=fixed_point_percentile,
                                         smooth_skel_sigma=smooth_skel_sigma,
                                         power_dist=power_dist))
            else:
                res.append(lazy_flow(labelled_array[sli], 
                                         dtform_method=dtform_method, 
                                         guide_img=None, 
                                         fixed_point_percentile=fixed_point_percentile,
                                         smooth_skel_sigma=smooth_skel_sigma,
                                         power_dist=power_dist))
        else:
            if guide_image is not None:
                res.append(lazy_flow(labelled_array[sli], 
                                         dtform_method=dtform_method, 
                                         guide_img=guide_image[sli], 
                                         fixed_point_percentile=fixed_point_percentile,
                                         smooth_skel_sigma=smooth_skel_sigma,
                                         iter_factor=iter_factor))
            else:
                res.append(lazy_flow(labelled_array[sli], 
                                         dtform_method=dtform_method, 
                                         guide_img=None, 
                                         fixed_point_percentile=fixed_point_percentile,
                                         smooth_skel_sigma=smooth_skel_sigma,
                                         power_dist=power_dist,
                                         iter_factor=iter_factor))
    res = dask.compute(res)
    res = np.array(res[0], dtype=np.float32)
         
    client.close()
    
    return res


# parallel above but for dtforms
def _distance_tform_labels2D_chunk(labelled_array, 
                                    dtform_fnc, clip=False):

     import numpy as np
     
     dtform = np.array([distance_tform_labels2D(labelled_array[zz], 
                                                dtform_fnc = dtform_fnc,
                                                clip=clip) for zz in np.arange(len(labelled_array))])
     
     return dtform
 
    
def distance_tform_labels2D_parallel(labelled_array, 
                                     dtform_fnc,
                                     clip=False, 
                                     n_processes=4):
    import multiprocessing as mp
    import numpy as np 
    
    n_proc = n_processes
    chunksize = len(labelled_array)//n_proc
    
    # chunk the data. 
    chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None
        chunks.append(labelled_array[slice(chunkstart, chunkend)])
 
    assert sum(map(len, chunks)) == len(labelled_array)
         
    # parallel computing 
    with mp.Pool(processes=n_proc) as pool:
        # starts the sub-processes without blocking
        proc_results = [pool.apply_async(_distance_tform_labels2D_chunk,
                                           args=(chunk, dtform_fnc, clip))
                                         for chunk in chunks]
       
        # blocks until all results are fetched
        res = np.vstack([r.get() for r in proc_results])
         
    return res



def _relabel_slices(labelled, bg_label=0):
    
    import numpy as np 
    max_ID = 0 
    labelled_ = []
    for lab in labelled:
        lab_ = lab.copy() # make a copy! 
        lab_[lab>bg_label] = lab[lab>bg_label] + max_ID # only update the foreground! 
        labelled_.append(lab_)
        if np.max(lab) > 0:
            max_ID = np.max(lab)+1
        
    labelled_ = np.array(labelled_)
    
    return labelled_


def remove_bad_flow_masks_2D(mask, 
                             flow, 
                             flow_threshold=0.6,
                             dtform_method='cellpose_improve',  
                             fixed_point_percentile=0.01, 
                             n_processes=4,
                             power_dist=None,
                             alpha=0.5, 
                             filter_scale = 1):
    """ 
    """
    import numpy as np 
    from scipy import ndimage
    from .filters import var_combine
    import skimage.measure as skmeasure 
    
    
    if flow.shape[1:] != mask.shape:
        print('ERROR: input ref flow is not same size as predicted masks')
        return
    
    if dtform_method=='cellpose':
        print('ERROR: not yet implemented.')
        return
    else:
        label_xy = mask.copy()
        guide_image_xy = distance_transform_labels_fast(label_xy)
        
        labels_gradients = distance_centroid_tform_flow_labels2D(label_xy, 
                                                                  dtform_method=dtform_method, 
                                                                  guide_img=guide_image_xy, 
                                                                  fixed_point_percentile=fixed_point_percentile, 
                                                                  iter_factor=5, 
                                                                  power_dist=power_dist)
        labels_gradients = labels_gradients[1:].copy()
        del label_xy, guide_image_xy
        
        labels_gradients = labels_gradients / (np.linalg.norm(labels_gradients, axis=0)[None,...] + 1e-20) # do a renormalize!. 


    print('finished computing gradients and removing bad flow masks ...')
    flow_errors=np.zeros(mask.max())
    for i in range(labels_gradients.shape[0]): # over the 3 dimensions. # this should be fast. 
        flow_errors += ndimage.mean((labels_gradients[i] - flow[i])**2, 
                                    mask,
                                    index=np.arange(1, mask.max()+1))
    flow_errors[np.isnan(flow_errors)] = 0 # zero out any nan entries!. 
        
    # 0.6 is the most stringent 
    error_labels = flow_errors > flow_threshold # ok this is quite essential postprocessing! and removes the lion's share of ill-formed masks at the borders!. 
    
    if np.sum(error_labels) > 0:
        reg_error_labels = np.arange(1, mask.max()+1)[error_labels] # these are the labels to remove!. 
        
        # set these regions to 0. 
        mask_new = mask.copy()
        mask_regionprops = skmeasure.regionprops(mask) # use this. 
        
        for lab in reg_error_labels: # so this need to be faster. but how ? 
            # mask_new[mask==lab] = 0
            coords = mask_regionprops[lab-1].coords
            mask_new[coords[:,0],
                     coords[:,1]] = 0
        return mask_new, flow_errors, labels_gradients # return out the intermediate information!. 
    else:
        return mask, flow_errors, labels_gradients



def remove_bad_flow_masks_3D(mask, 
                             flow, 
                             flow_threshold=0.6,
                             dtform_method='cellpose_improve',  
                             fixed_point_percentile=0.01, 
                             n_processes=4,
                             smooth_skel_sigma=3,
                             power_dist=None,
                             alpha=0.5, 
                             smooth_sigma=0,
                             smooth_ds=1,
                             filter_scale = 1):
    """ 
    """
    import numpy as np 
    from scipy import ndimage
    from .filters import var_combine, smooth_vol
    import skimage.measure as skmeasure 
    
    
    if flow.shape[1:] != mask.shape:
        print('ERROR: input ref flow is not same size as predicted masks')
        return
    # # maybe don't need? # remove this!. 
    # label_xy = _relabel_slices(mask)
    # label_xz = _relabel_slices(mask.transpose(1,0,2))
    # label_yz = _relabel_slices(mask.transpose(2,0,1))

    # label_xy = mask.copy()
    # label_xz = mask.transpose(1,0,2)
    # label_yz = mask.transpose(2,0,1)
    
    # import pylab as plt 
    # plt.figure()
    # plt.imshow(label_xy[100])
    # plt.show(block=False)
    
    # plt.figure()
    # plt.imshow(label_xz[label_xz.shape[0]//2])
    # plt.show(block=False)

    if dtform_method=='cellpose':
        print('ERROR: not yet implemented.')
        return
    else:
        print('starting xy')
        label_xy = mask.copy()
        guide_image_xy = distance_transform_labels_fast(label_xy)
        # mask_xy_gradient = distance_centroid_tform_flow_labels2D_parallel(label_xy, 
        #                                                                           dtform_method=dtform_method,
        #                                                                           guide_image=guide_image_xy,
        #                                                                           fixed_point_percentile=fixed_point_percentile, 
        #                                                                           n_processes=n_processes,
        #                                                                           power_dist=power_dist)
        mask_xy_gradient = distance_centroid_tform_flow_labels2D_dask(label_xy, 
                                                                        dtform_method=dtform_method,
                                                                        guide_image=guide_image_xy,
                                                                        # guide_image = None,
                                                                        fixed_point_percentile=fixed_point_percentile, 
                                                                        n_processes=n_processes,
                                                                        smooth_skel_sigma=smooth_skel_sigma,
                                                                        power_dist=power_dist)
        mask_xy_gradient = mask_xy_gradient[:,1:].copy()
        mask_xy_gradient = np.concatenate([np.zeros_like(mask_xy_gradient[:,1])[:,None,...], mask_xy_gradient], axis=1)
        mask_xy_gradient = mask_xy_gradient[:,[0,1,2],...] #.copy() # we must flip the channels!. 
        
        
        print('done xy ')
        del label_xy, guide_image_xy
        
        label_xz = mask.transpose(1,0,2)
        guide_image_xz = distance_transform_labels_fast(label_xz)
        # mask_xz_gradient = distance_centroid_tform_flow_labels2D_parallel(label_xz, 
        #                                                                         dtform_method=dtform_method,
        #                                                                           guide_image=guide_image_xz,
        #                                                                           fixed_point_percentile=fixed_point_percentile, 
        #                                                                           n_processes=n_processes,
        #                                                                           power_dist=power_dist)
        mask_xz_gradient = distance_centroid_tform_flow_labels2D_dask(label_xz, 
                                                                        dtform_method=dtform_method,
                                                                           guide_image=guide_image_xz,
                                                                          # guide_image = None, 
                                                                          fixed_point_percentile=fixed_point_percentile, 
                                                                          n_processes=n_processes,
                                                                          smooth_skel_sigma=smooth_skel_sigma,
                                                                          power_dist=power_dist)
        mask_xz_gradient = mask_xz_gradient[:,1:].copy()
        mask_xz_gradient = mask_xz_gradient.transpose( 2,1,0,3 )
        mask_xz_gradient = np.concatenate([np.zeros_like(mask_xz_gradient[:,1])[:,None,...], mask_xz_gradient], axis=1)
        mask_xz_gradient = mask_xz_gradient[:,[1,0,2],...]#.copy() # we must flip the channels!. 
        
        print('done xz')
        del label_xz, guide_image_xz
        
        label_yz = mask.transpose(2,0,1)
        guide_image_yz = distance_transform_labels_fast(label_yz)
        # mask_yz_gradient = distance_centroid_tform_flow_labels2D_parallel(label_yz, 
        #                                                                         dtform_method=dtform_method,
        #                                                                           guide_image=guide_image_yz,
        #                                                                           fixed_point_percentile=fixed_point_percentile, 
        #                                                                           n_processes=n_processes,
        #                                                                           power_dist=power_dist)
        mask_yz_gradient = distance_centroid_tform_flow_labels2D_dask(label_yz, 
                                                                        dtform_method=dtform_method,
                                                                           guide_image=guide_image_yz,
                                                                          # guide_image = None,
                                                                          fixed_point_percentile=fixed_point_percentile, 
                                                                          n_processes=n_processes,
                                                                          smooth_skel_sigma=smooth_skel_sigma,
                                                                          power_dist=power_dist)
        mask_yz_gradient = mask_yz_gradient[:,1:].copy()
        mask_yz_gradient = mask_yz_gradient.transpose(2,1,3,0) 
        mask_yz_gradient = np.concatenate([np.zeros_like(mask_yz_gradient[:,1])[:,None,...], mask_yz_gradient], axis=1)
        mask_yz_gradient = mask_yz_gradient[:,[1,2,0],...] #.copy() # we must flip the channels!. 
        
        print('done yz')
        del label_yz, guide_image_yz
        
    
        dx = var_combine([ndimage.gaussian_filter(mask_xy_gradient[:,2], sigma=1), 
                                        ndimage.gaussian_filter(mask_xz_gradient[:,2], sigma=1)],
                                      ksize=filter_scale,
                                      alpha=0.5)
        dy = var_combine([ndimage.gaussian_filter(mask_xy_gradient[:,1],sigma=1), 
                                        ndimage.gaussian_filter(mask_yz_gradient[:,1],sigma=1)],
                                      ksize=filter_scale, # we probably need to tune this up! for this imaging 
                                      alpha=0.5)
        dz = var_combine([ndimage.gaussian_filter(mask_xz_gradient[:,0], sigma=1), 
                                        ndimage.gaussian_filter(mask_yz_gradient[:,0], sigma=1)],
                                      ksize=filter_scale,
                                      alpha=0.5)
        
        
        # does this change ..... add less smoothing!. 
        dx = ndimage.gaussian_filter(dx, sigma=1.)
        dy = ndimage.gaussian_filter(dy, sigma=1.)
        dz = ndimage.gaussian_filter(dz, sigma=1.)
        
        labels_gradients = np.concatenate([ dz[:,None,...], 
                                            dy[:,None,...], 
                                            dx[:,None,...] ], axis=1)
        del dx, dy, dz
        
        # do a little filtering. 
        labels_gradients = labels_gradients.transpose(1,0,2,3)
        
        print('computed true gradients', labels_gradients.shape)
        
        if smooth_sigma > 0: 
            # do the additional smoothing!. 
            for ch_no in np.arange(len(labels_gradients)):
                labels_gradients[ch_no] = smooth_vol(labels_gradients[ch_no], ds=smooth_ds, smooth=smooth_sigma)
        # else:

        labels_gradients = labels_gradients / (np.linalg.norm(labels_gradients, axis=0)[None,...] + 1e-20) # do a renormalize!. 


    print('finished computing gradients and removing bad flow masks ...')
    flow_errors=np.zeros(mask.max())
    for i in range(labels_gradients.shape[0]): # over the 3 dimensions. # this should be fast. 
        flow_errors += ndimage.mean((labels_gradients[i] - flow[i])**2, 
                                    mask,
                                    index=np.arange(1, mask.max()+1))
    flow_errors[np.isnan(flow_errors)] = 0 # zero out any nan entries!. 
        
    # 0.6 is the most stringent 
    error_labels = flow_errors > flow_threshold # ok this is quite essential postprocessing! and removes the lion's share of ill-formed masks at the borders!. 
    
    
    if np.sum(error_labels) > 0:
        
        # we need this to not encounter errors using skmeasure.regionprops
        valid_labels = np.setdiff1d(np.unique(mask), 0)
        reg_error_labels = np.arange(1, mask.max()+1)[error_labels] # these are the labels to remove!. 
        
        # print('intersecting...')
        reg_error_labels = np.intersect1d(reg_error_labels, valid_labels) # these must be in the intersection. 
        # ok ... this is still erroring because we have to be able to mtach the index. 
        
        # match the indices. 
        reg_error_labels_indices = np.hstack([np.arange(len(valid_labels))[valid_labels==ll] for ll in reg_error_labels])
        
        
        # set these regions to 0. 
        mask_new = mask.copy()
        mask_regionprops = skmeasure.regionprops(mask) # use this. -- # the problem here is this skips.
        
        # print(len(mask_regionprops))
        # print(reg_error_labels)
                
        # for lab in reg_error_labels: # so this need to be faster. but how ?
        for lab in reg_error_labels_indices:
            # # mask_new[mask==lab] = 0
            # coords = mask_regionprops[lab-1].coords # was lab-1  # this is wrong? --- i don't have as many of these. 
            coords = mask_regionprops[lab].coords
            mask_new[coords[:,0],
                     coords[:,1],
                     coords[:,2]] = 0
        return mask_new, flow_errors, labels_gradients # return out the intermediate information!. 
    else:
        return mask, flow_errors, labels_gradients
    
    
def remove_bad_flow_masks_3D_edt_from2D(mask, 
                                         flow, 
                                         flow_threshold=0.6,
                                         fixed_point_percentile=0.01, 
                                         n_processes=4,
                                         power_dist=None,
                                         alpha=0.5, 
                                         filter_scale = 1):
    """ 
    """
    import numpy as np 
    from scipy import ndimage
    from .filters import var_combine
    import skimage.measure as skmeasure 
    
    
    if flow.shape[1:] != mask.shape:
        print('ERROR: input ref flow is not same size as predicted masks')
        return
    
    label_xy = mask.copy()

    flows_xy_dist = distance_transform_labels_fast(label_xy)
    flows_xy = np.array([np.array(np.gradient(dist_xy_slice)) for dist_xy_slice in flows_xy_dist])
    flows_xy = flows_xy / (np.linalg.norm(flows_xy, axis=1)[:,None] + 1e-20)

    flows_xy = np.concatenate([np.zeros_like(flows_xy[:,1])[:,None,...], flows_xy], axis=1)
     
    print('done xy ')
    del label_xy
    
    label_xz = mask.transpose(1,0,2)
    
    flows_xz_dist = distance_transform_labels_fast(label_xz)
    flows_xz = np.array([np.array(np.gradient(dist_xz_slice)) for dist_xz_slice in flows_xz_dist])
    flows_xz = flows_xz / (np.linalg.norm(flows_xz, axis=1)[:,None] + 1e-20)
    
    flows_xz = flows_xz.transpose(2,1,0,3)
    flows_xz = np.concatenate([np.zeros_like(flows_xz[:,1])[:,None,...], flows_xz], axis=1)
    
    print('done xz')
    del label_xz
    
    label_yz = mask.transpose(2,0,1)
    flows_yz_dist = distance_transform_labels_fast(label_yz*1)
    flows_yz = np.array([np.array(np.gradient(dist_yz_slice)) for dist_yz_slice in flows_yz_dist])
    flows_yz = flows_yz / (np.linalg.norm(flows_yz, axis=1)[:,None] + 1e-20)
    
    flows_yz = flows_yz.transpose(2,1,3,0)
    flows_yz = np.concatenate([np.zeros_like(flows_yz[:,1])[:,None,...], flows_yz], axis=1)
    
    print('done yz')
    del label_yz
    
    mask_xy_gradient = flows_xy[:,[0,1,2],...].copy()
    mask_xz_gradient = flows_xz[:,[1,0,2],...].copy()
    mask_yz_gradient = flows_yz[:,[1,2,0]].copy()


    dx = var_combine([ndimage.gaussian_filter(mask_xy_gradient[:,2], sigma=1), 
                                    ndimage.gaussian_filter(mask_xz_gradient[:,2], sigma=1)],
                                  ksize=filter_scale,
                                  alpha=0.5)
    dy = var_combine([ndimage.gaussian_filter(mask_xy_gradient[:,1],sigma=1), 
                                    ndimage.gaussian_filter(mask_yz_gradient[:,1],sigma=1)],
                                  ksize=filter_scale, # we probably need to tune this up! for this imaging 
                                  alpha=0.5)
    dz = var_combine([ndimage.gaussian_filter(mask_xz_gradient[:,0], sigma=1), 
                                    ndimage.gaussian_filter(mask_yz_gradient[:,0], sigma=1)],
                                  ksize=filter_scale,
                                  alpha=0.5)
    
    labels_gradients = np.concatenate([ dz[:,None,...], 
                                        dy[:,None,...], 
                                        dx[:,None,...] ], axis=1)
    del dx, dy, dz
    
    # do a little filtering. 
    labels_gradients = labels_gradients.transpose(1,0,2,3)
    labels_gradients = labels_gradients / (np.linalg.norm(labels_gradients, axis=0)[None,...] + 1e-20) # do a renormalize!. 

    print(labels_gradients.shape)

    print('finished computing gradients and removing bad flow masks ...')
    flow_errors=np.zeros(mask.max())
    for i in range(labels_gradients.shape[0]): # over the 3 dimensions. # this should be fast. 
        flow_errors += ndimage.mean((labels_gradients[i] - flow[i])**2, 
                                    mask,
                                    index=np.arange(1, mask.max()+1))
    flow_errors[np.isnan(flow_errors)] = 0 # zero out any nan entries!. 
        
    # 0.6 is the most stringent 
    error_labels = flow_errors > flow_threshold # ok this is quite essential postprocessing! and removes the lion's share of ill-formed masks at the borders!. 
    
    if np.sum(error_labels) > 0:
        reg_error_labels = np.arange(1, mask.max()+1)[error_labels] # these are the labels to remove!. 
        
        # set these regions to 0. 
        mask_new = mask.copy()
        mask_regionprops = skmeasure.regionprops(mask) # use this. 
        
        for lab in reg_error_labels: # so this need to be faster. but how ? 
            # mask_new[mask==lab] = 0
            coords = mask_regionprops[lab-1].coords
            mask_new[coords[:,0],
                     coords[:,1],
                     coords[:,2]] = 0
        return mask_new, flow_errors, labels_gradients # return out the intermediate information!. 
    else:
        return mask, flow_errors, labels_gradients
    
    
def remove_bad_flow_masks_3D_edt(mask, 
                                 flow, 
                                 flow_threshold=0.6,
                                 dtform_method='cellpose_improve',  
                                 fixed_point_percentile=0.01, 
                                 n_processes=4,
                                 power_dist=None,
                                 alpha=0.5, 
                                 filter_scale = 1):
    """ 
    """
    import numpy as np 
    from scipy import ndimage
    from .filters import var_combine
    import skimage.measure as skmeasure 
    import edt 
    
    
    if flow.shape[1:] != mask.shape:
        print('ERROR: input ref flow is not same size as predicted masks')
        return

    if dtform_method=='cellpose':
        print('ERROR: not yet implemented.')
        return
    else:
        
    
        edt_3D = edt.edt(mask, 
                         black_border=True, 
                         order='C', 
                         parallel=n_processes).astype(np.float32)       
        
        labels_gradients = np.array(np.gradient(edt_3D))
        
        # # do a little filtering. 
        # labels_gradients = labels_gradients.transpose(1,0,2,3)
        labels_gradients = labels_gradients / (np.linalg.norm(labels_gradients, axis=0)[None,...] + 1e-20) # do a renormalize!. 


    print('finished computing gradients and removing bad flow masks ...')
    flow_errors=np.zeros(mask.max())
    for i in range(labels_gradients.shape[0]): # over the 3 dimensions. # this should be fast. 
        flow_errors += ndimage.mean((labels_gradients[i] - flow[i])**2, 
                                    mask,
                                    index=np.arange(1, mask.max()+1))
    flow_errors[np.isnan(flow_errors)] = 0 # zero out any nan entries!. 
        
    # 0.6 is the most stringent 
    error_labels = flow_errors > flow_threshold # ok this is quite essential postprocessing! and removes the lion's share of ill-formed masks at the borders!. 
    
    if np.sum(error_labels) > 0:
        reg_error_labels = np.arange(1, mask.max()+1)[error_labels] # these are the labels to remove!. 
        
        # set these regions to 0. 
        mask_new = mask.copy()
        mask_regionprops = skmeasure.regionprops(mask) # use this. 
        
        for lab in reg_error_labels: # so this need to be faster. but how ? 
            # mask_new[mask==lab] = 0
            coords = mask_regionprops[lab-1].coords
            mask_new[coords[:,0],
                     coords[:,1],
                     coords[:,2]] = 0
        return mask_new, flow_errors, labels_gradients # return out the intermediate information!. 
    else:
        return mask, flow_errors, labels_gradients
    
    
def remove_bad_flow_masks_3D_mp(mask, 
                                 flow, 
                                 flow_threshold=0.6,
                                 dtform_method='cellpose_improve',  
                                 fixed_point_percentile=0.01, 
                                 n_processes=4,
                                 power_dist=None,
                                 alpha=0.5, 
                                 filter_scale = 1):
    """ 
    """
    import numpy as np 
    from scipy import ndimage
    from .filters import var_combine
    import skimage.measure as skmeasure 
    import multiprocess as mp 
    from multiprocess.pool import ThreadPool
    
    mp.set_start_method('spawn')
    
    
    if flow.shape[1:] != mask.shape:
        print('ERROR: input ref flow is not same size as predicted masks')
        return
    
    if dtform_method=='cellpose':
        print('ERROR: not yet implemented.')
        return
    else:
        
        tmp = mask.copy()
        guide = distance_transform_labels_fast(tmp, n_threads=n_processes)
        
        # define this internal function for multiprocessing pool to operate on 
        def _compute_cellflow_view(slice_id):
            # out = grad_flows.distance_centroid_tform_flow_labels2D()
            out = distance_centroid_tform_flow_labels2D(tmp[slice_id], 
                                                        dtform_method=dtform_method, 
                                                        guide_img=guide[slice_id], 
                                                        fixed_point_percentile=fixed_point_percentile, 
                                                        power_dist=power_dist)
            
            return out 
        
        print('starting xy')
        with mp.Pool(n_processes) as pool: 
            mask_xy_gradient = pool.map(_compute_cellflow_view, range(0, len(tmp)))
            mask_xy_gradient = np.array(mask_xy_gradient)
        
        mask_xy_gradient = mask_xy_gradient[:,1:]
        mask_xy_gradient = np.concatenate([np.zeros_like(mask_xy_gradient[:,1])[:,None,...], mask_xy_gradient], axis=1)
        mask_xy_gradient = mask_xy_gradient[:,[0,1,2],...] #.copy() # we must flip the channels!. 

        print('done xy ')
        
        tmp = mask.transpose(1,0,2).copy()
        guide = distance_transform_labels_fast(tmp, n_threads=n_processes)
        
        print('starting xz')
        with mp.Pool(n_processes) as pool: 
            mask_xz_gradient = pool.map(_compute_cellflow_view, range(0, len(tmp)))
            mask_xz_gradient = np.array(mask_xz_gradient)
        
        mask_xz_gradient = mask_xz_gradient[:,1:].copy()
        mask_xz_gradient = mask_xz_gradient.transpose( 2,1,0,3 )
        mask_xz_gradient = np.concatenate([np.zeros_like(mask_xz_gradient[:,1])[:,None,...], mask_xz_gradient], axis=1)
        mask_xz_gradient = mask_xz_gradient[:,[1,0,2],...]#.copy() # we must flip the channels!. 
        print('done xz')
        
        
        tmp = mask.transpose(2,0,1).copy()
        guide = distance_transform_labels_fast(tmp, n_threads=n_processes)
        
        print('starting yz')
        with mp.Pool(n_processes) as pool: 
            mask_yz_gradient = pool.map(_compute_cellflow_view, range(0, len(tmp)))
            mask_yz_gradient = np.array(mask_yz_gradient)
        
        mask_yz_gradient = mask_yz_gradient[:,1:].copy()
        mask_yz_gradient = mask_yz_gradient.transpose(2,1,3,0) 
        mask_yz_gradient = np.concatenate([np.zeros_like(mask_yz_gradient[:,1])[:,None,...], mask_yz_gradient], axis=1)
        mask_yz_gradient = mask_yz_gradient[:,[1,2,0],...] #.copy() # we must flip the channels!. 
        
        print('done yz')
        del tmp, guide
            
        dx = var_combine([ndimage.gaussian_filter(mask_xy_gradient[:,2], sigma=1), 
                                        ndimage.gaussian_filter(mask_xz_gradient[:,2], sigma=1)],
                                      ksize=filter_scale,
                                      alpha=0.5)
        dy = var_combine([ndimage.gaussian_filter(mask_xy_gradient[:,1],sigma=1), 
                                        ndimage.gaussian_filter(mask_yz_gradient[:,1],sigma=1)],
                                      ksize=filter_scale, # we probably need to tune this up! for this imaging 
                                      alpha=0.5)
        dz = var_combine([ndimage.gaussian_filter(mask_xz_gradient[:,0], sigma=1), 
                                        ndimage.gaussian_filter(mask_yz_gradient[:,0], sigma=1)],
                                      ksize=filter_scale,
                                      alpha=0.5)
        
        labels_gradients = np.concatenate([ dz[:,None,...], 
                                            dy[:,None,...], 
                                            dx[:,None,...] ], axis=1)
        del dx, dy, dz
        
        # do a little filtering. 
        labels_gradients = labels_gradients.transpose(1,0,2,3)
        labels_gradients = labels_gradients / (np.linalg.norm(labels_gradients, axis=0)[None,...] + 1e-20) # do a renormalize!. 


    print('finished computing gradients and removing bad flow masks ...')
    flow_errors=np.zeros(mask.max())
    for i in range(labels_gradients.shape[0]): # over the 3 dimensions. # this should be fast. 
        flow_errors += ndimage.mean((labels_gradients[i] - flow[i])**2, 
                                    mask,
                                    index=np.arange(1, mask.max()+1))
    flow_errors[np.isnan(flow_errors)] = 0 # zero out any nan entries!. 
        
    # 0.6 is the most stringent 
    error_labels = flow_errors > flow_threshold # ok this is quite essential postprocessing! and removes the lion's share of ill-formed masks at the borders!. 
    
    if np.sum(error_labels) > 0:
        reg_error_labels = np.arange(1, mask.max()+1)[error_labels] # these are the labels to remove!. 
        
        # set these regions to 0. 
        mask_new = mask.copy()
        mask_regionprops = skmeasure.regionprops(mask) # use this. 
        
        for lab in reg_error_labels: # so this need to be faster. but how ? 
            # mask_new[mask==lab] = 0
            coords = mask_regionprops[lab-1].coords
            mask_new[coords[:,0],
                     coords[:,1],
                     coords[:,2]] = 0
        return mask_new, flow_errors, labels_gradients # return out the intermediate information!. 
    else:
        return mask, flow_errors, labels_gradients

