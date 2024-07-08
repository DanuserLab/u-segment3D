# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:35:23 2023

@author: fyz11
"""

def _mkdir(directory):
    
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return []

def _normalize99(Y,lower=0.01,upper=99.99):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile 
    Upper and lower percentile ranges configurable. 
    
    Parameters
    ----------
    Y: ndarray, float
        Component array of lenth N by L1 by L2 by ... by LN. 
    upper: float
        upper percentile above which pixels are sent to 1.0
    
    lower: float
        lower percentile below which pixels are sent to 0.0
    
    Returns
    --------------
    normalized array with a minimum of 0 and maximum of 1
    
    """
    import numpy as np
    
    X = Y.copy()
    
    return np.interp(X, (np.percentile(X, lower), np.percentile(X, upper)), (0, 1))


def _interp2(query_pts, grid_shape, I_ref, method='linear', cast_uint8=False):
    
    import numpy as np 
    from scipy.interpolate import RegularGridInterpolator 
    
    spl = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                   np.arange(grid_shape[1])), 
                                   I_ref, method=method, bounds_error=False, fill_value=0)
    I_query = spl((query_pts[...,0], 
                   query_pts[...,1]))

    if cast_uint8:
        I_query = np.uint8(I_query)
    
    return I_query
    
def _interp3(query_pts, grid_shape, I_ref, method='linear', cast_uint8=False):
    
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np 
    
    spl_3 = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                         np.arange(grid_shape[1]), 
                                         np.arange(grid_shape[2])), 
                                         I_ref, method=method, bounds_error=False, fill_value=0)
    
    I_query = spl_3((query_pts[...,0], 
                      query_pts[...,1],
                      query_pts[...,2]))
    if cast_uint8:
        I_query = np.uint8(I_query)
    
    return I_query

# =============================================================================
# 2D stuff 
# =============================================================================
def connected_components_pts_2D( pts, pts0, shape, 
                                smooth_sigma=1, 
                                thresh_factor=None, 
                                mask=None,
                                min_area=1) : 

    import numpy as np 
    import scipy.ndimage as ndimage
    import skimage.measure as skmeasure
    import skimage.segmentation as sksegmentation 
    
    # parse ... 
    votes_grid_acc = np.zeros(shape)
    
    # count
    votes_grid_acc[(pts[:,0]).astype(np.int32), 
                   (pts[:,1]).astype(np.int32)] += 1. # add a vote. 
                   
    # smooth to get a density (fast KDE estimation)
    votes_grid_acc = ndimage.gaussian_filter(votes_grid_acc, sigma=smooth_sigma)  
    
    if thresh_factor is not None:
        if mask is not None:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc[mask]) + thresh_factor*np.std(votes_grid_acc[mask])
        else:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc) + thresh_factor*np.std(votes_grid_acc)
    else:
        votes_grid_binary = votes_grid_acc > np.mean(votes_grid_acc) # just threshold over the mean. 
        
    cell_seg_connected = skmeasure.label(votes_grid_binary, connectivity=1) # use the full conditional 
    cell_uniq_regions = np.setdiff1d(np.unique(cell_seg_connected),0)
    if len(cell_uniq_regions)>0:
        props = skmeasure.regionprops(cell_seg_connected)
        areas = np.hstack([re.area for re in props])
        invalid_areas = cell_uniq_regions[areas<=min_area]
    
        for invalid in invalid_areas:
            cell_seg_connected[cell_seg_connected==invalid] = 0
        
    if cell_seg_connected.max() > 0:
        cell_seg_connected = sksegmentation.relabel_sequential(cell_seg_connected)[0]
    
    
    cell_seg_connected_original = np.zeros_like(cell_seg_connected)
    cell_seg_connected_original[(pts0[:,0]).astype(np.int32), 
                                (pts0[:,1]).astype(np.int32)] = cell_seg_connected[(pts[:,0]).astype(np.int32), 
                                                                                   (pts[:,1]).astype(np.int32)]
    
    if mask is not None:
        cell_seg_connected[mask == 0] = 0
        cell_seg_connected_original[mask==0] = 0 # also mask the predicted. 

    return cell_seg_connected_original, cell_seg_connected, votes_grid_acc # return the accumulator.!    


def connected_components_pts_3D( pts, pts0, shape, 
                                smooth_sigma=1, 
                                thresh_factor=None, 
                                mask=None,
                                min_area=1) : 

    import numpy as np 
    import scipy.ndimage as ndimage
    import skimage.measure as skmeasure
    import skimage.segmentation as sksegmentation 
    import cc3d
    
    # parse ... 
    votes_grid_acc = np.zeros(shape)
    
    # count
    votes_grid_acc[(pts[:,0]).astype(np.int32), 
                   (pts[:,1]).astype(np.int32),
                   (pts[:,2]).astype(np.int32)] += 1. # add a vote. 
                   
    # smooth to get a density (fast KDE estimation)
    votes_grid_acc = ndimage.gaussian_filter(votes_grid_acc, sigma=smooth_sigma)  
    
    if thresh_factor is not None:
        if mask is not None:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc[mask]) + thresh_factor*np.std(votes_grid_acc[mask])
        else:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc) + thresh_factor*np.std(votes_grid_acc)
    else:
        votes_grid_binary = votes_grid_acc > np.mean(votes_grid_acc) # just threshold over the mean. 
        
    # cell_seg_connected = skmeasure.label(votes_grid_binary, connectivity=2) # use the full conditional 
    cell_seg_connected = cc3d.connected_components(votes_grid_binary)
    
    # # we don't need this bit. 
    # cell_uniq_regions = np.setdiff1d(np.unique(cell_seg_connected),0)
    # if len(cell_uniq_regions)>0:
    #     props = skmeasure.regionprops(cell_seg_connected)
    #     areas = np.hstack([re.area for re in props])
    #     invalid_areas = cell_uniq_regions[areas<=min_area]
    
    #     for invalid in invalid_areas:
    #         cell_seg_connected[cell_seg_connected==invalid] = 0
        
    # if cell_seg_connected.max() > 0:
    #     cell_seg_connected = sksegmentation.relabel_sequential(cell_seg_connected)[0]
    
    cell_seg_connected_original = np.zeros_like(cell_seg_connected)
    cell_seg_connected_original[(pts0[:,0]).astype(np.int32), 
                                (pts0[:,1]).astype(np.int32),
                                (pts0[:,2]).astype(np.int32)] = cell_seg_connected[(pts[:,0]).astype(np.int32), 
                                                                                 (pts[:,1]).astype(np.int32),
                                                                                 (pts[:,2]).astype(np.int32)]
    
    if mask is not None:
        cell_seg_connected[mask == 0] = 0
        cell_seg_connected_original[mask==0] = 0 # also mask the predicted. 

    return cell_seg_connected_original, cell_seg_connected, votes_grid_acc # return the accumulator.!    


def _sdf_distance_transform(binary, rev_sign=True): 
    
    import numpy as np 
    from scipy.ndimage import distance_transform_edt
    # import skfmm
    # import GeodisTK
    
    pos_binary = binary.copy()
    neg_binary = np.logical_not(pos_binary)
    
    res = distance_transform_edt(neg_binary) * neg_binary - (distance_transform_edt(pos_binary) - 1) * pos_binary
    # res = skfmm.distance(neg_binary, dx=0.5) * neg_binary - (skfmm.distance(pos_binary, dx=0.5) - 1) * pos_binary
    # res = skfmm.distance(neg_binary) * neg_binary - (skfmm.distance(pos_binary) - 1) * pos_binary # this was fast!. 
    # res = geodesic_distance_2d((neg_binary*1.).astype(np.float32), S=neg_binary, lamb=0.8, iter=10) * neg_binary - (geodesic_distance_2d((pos_binary*1.).astype(np.float32), S=neg_binary, lamb=0.5, iter=10) - 1) * pos_binary
    
    if rev_sign:
        res = res * -1
    
    return res



def surf_normal_sdf(binary, return_sdf=True, smooth_gradient=None, eps=1e-12, norm_vectors=True):

    import numpy as np 
    import scipy.ndimage as ndimage

    sdf_vol = _sdf_distance_transform(binary, rev_sign=True) # so that we have it pointing outwards!. 
    
    # compute surface normal of the signed distance function. 
    sdf_vol_normal = np.array(np.gradient(sdf_vol))
    # smooth gradient
    if smooth_gradient is not None: # smoothing needs to be done before normalization of magnitude. 
        sdf_vol_normal = np.array([ndimage.gaussian_filter(sdf, sigma=smooth_gradient) for sdf in sdf_vol_normal])

    if norm_vectors:
        sdf_vol_normal = sdf_vol_normal / (np.linalg.norm(sdf_vol_normal, axis=0)[None,:]+eps)

    return sdf_vol_normal, sdf_vol


def mean_curvature_sdf(sdf_normal):

    def divergence(f):
        import numpy as np 
        """
        Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
        :param f: List of ndarrays, where every item of the list is one dimension of the vector field
        :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
        """
        num_dims = len(f)
        return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])
        
    H = .5*(divergence(sdf_normal))# total curvature is the divergence of the normal. 
    
    return H 


def gradient_watershed2D_binary(binary, 
                                gradient_img=None, 
                                momenta = 0.0,
                                divergence_rescale=True, 
                                smooth_sigma=1, 
                                smooth_gradient=1, 
                                delta=.5, 
                                n_iter=10, 
                                min_area=5, 
                                gradient_decay=0.0,
                                eps=1e-20, 
                                interp=True,
                                thresh_factor=None, 
                                track_flow=True, # if track_flow then we record!. 
                                mask=None,
                                binary_mask_gradient=False,
                                debug_viz=False):
    
    """ parses the instance level segmentation implicitly given as an input binary or a vector field. 
    The algorithm works as an inverse watershed.
    
    Step 1: a grid of points is seeds on the image
    Step 2: points are propagated for n_iter according to the gradient_img, condensing towards cell centers implicitly implied by the gradient image.
    Step 3: individual cluster centers are found by binarisation and connected component, removing objects < min_area
    
    result is an integer image the same size as binary. 

    Parameters
    ----------
    binary : (MxNxL) numpy array
        input binary image defining the voxels that need labeling
    gradient_img :  (MxNxLx3) numpy array
        This is a gradient field such as that from applying np.array(np.gradient(img)).transpose(1,2,3,0) where img is a potential such as a distance transform or probability map. 
    divergence_rescale : 
        If True, the gradient_img is scaled by the divergence which is equivalent to the mean curvature, this helps to prevent early breakage for tube-like structures.   
    smooth_sigma : scalar
        controls the catchment area for identifying distinct cells at the final propagation position. Smaller smooth_sigma leads to more oversegmentation. 
    smooth_gradient : scalar
        the isotropic sigma value controlling the Gaussian smoothing of the gradient field. More smoothing results in more cells grouped together
    delta: scalar
        the voxel size to propagate grid points per iteration. Related to the stability. If too small takes too long. If too large, might not converge. if delta=1, takes a 1 voxel step. 
    n_iter: int 
        the number of iterations to run. (To do: monitor convergence and break early to improve speed)
    min_area: scalar
        volume of cells < min_area are removed. 
    eps: float
        a small number for numerical stability
    thresh_factor: scalar
        The final cells are identified by thresholding on a threshold mean+thresh_factor*std. Thresh_factor controls what is an object prior to connected components analysis 
    mask: (MxNxL) numpy array
        optional binary mask to gate the region to parse labels for.
    binary_mask_gradient: bool
        if True, multiply the gradient to zero out all gradients outside of the foreground binary
    debug_viz: bool
        if True, visualise the position of the points at every algorithm iteration. 
        
    Returns
    -------
    cell_seg_connected_original : (MxNxL)
        an integer image where each unique int > 0 relates to a unique object such that object 1 is retrieved by cell_seg_connected_original==1.
        
    """
    import scipy.ndimage as ndimage
    import numpy as np 
    import skimage.morphology as skmorph
    import pylab as plt 
    import skimage.measure as skmeasure 
    import skimage.segmentation as sksegmentation 
    from tqdm import tqdm 
    
    # compute the signed distance transform
    if gradient_img is not None:
        sdf_normals = gradient_img.transpose(2,0,1) # use the supplied gradients! 
        if binary_mask_gradient:
            sdf_normals = sdf_normals #* binary[None,...]
    else:
        sdf_normals, sdf_binary = surf_normal_sdf(binary, return_sdf=True, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True)
        if binary_mask_gradient:
            sdf_normals = sdf_normals #* binary[None,...]
        
    if divergence_rescale:
        # rescale the speed
        curvature_2D = mean_curvature_sdf(sdf_normals/(np.linalg.norm(sdf_normals, axis=0)[None,...]+eps))
        curvature_2D = _normalize99(curvature_2D) # rescales to a factor between 0-1
        sdf_normals = sdf_normals * curvature_2D[None,...] # multiplicative factor rescaling 
        
        
    # print(sdf_normals.shape)
    grid =  np.zeros(binary.shape, dtype=np.int32)
    pts = np.argwhere(binary>0) # (N,ndim)
    
    mu = momenta # momentum
    # g0 = np.zeros(pts.shape)
    g0 = np.zeros_like(pts)
    
    tracks = [pts]
    
    for ii in tqdm(np.arange(n_iter)):
        pt_ii = tracks[-1].copy()
        
        if interp:
            pts_vect_ii = np.array([_interp2(pt_ii, binary.shape, I_ref=sdf_normals[ch], method='linear', cast_uint8=False) for ch in np.arange(len(sdf_normals))]).T
        else:
            pts_vect_ii = sdf_normals[:,np.rint(pt_ii[:,0]).astype(np.int32), np.rint(pt_ii[:,1]).astype(np.int32)].T
        
        pts_vect_ii = pts_vect_ii / (np.linalg.norm(pts_vect_ii, axis=-1)[:,None] + eps)
        # pt_ii_next = pt_ii + delta*pts_vect_ii
        pt_ii_next = pt_ii + (delta*pts_vect_ii + mu*g0) * (1./(delta+mu))   * (delta/(1.+ii*gradient_decay)) # add momentum + connectivity. 
        g0 = pts_vect_ii.copy() # copy this into the past history....     
        
        pt_ii_next[:,0] = np.clip(pt_ii_next[:,0], 0, binary.shape[0]-1)
        pt_ii_next[:,1] = np.clip(pt_ii_next[:,1], 0, binary.shape[1]-1)
        
        if track_flow:
            tracks.append(pt_ii_next)
        else:
            tracks[-1] = pt_ii_next.copy() # copy over. 
        
        if debug_viz:
            plt.figure(figsize=(10,10))
            plt.imshow(binary)
            plt.plot(pt_ii_next[:,1], pt_ii_next[:,0], 'r.')
            plt.show(block=False)
        
    tracks = np.array(tracks)
    
    cell_seg_connected_original, cell_seg_connected, votes_grid_acc = connected_components_pts_2D( pts=tracks[-1], 
                                                                                                    pts0=pts, 
                                                                                                    shape=binary.shape[:2], 
                                                                                                    smooth_sigma=smooth_sigma, 
                                                                                                    thresh_factor=thresh_factor, 
                                                                                                    mask=mask,
                                                                                                    min_area=min_area)

    return cell_seg_connected_original, cell_seg_connected, tracks, votes_grid_acc


# can we make this v. fast? ---> it is mainly the advection that is slow....., if we could parallelize in blocks? 
def gradient_watershed3D_binary(binary, 
                                gradient_img=None, 
                                momenta = 0.75,
                                divergence_rescale=True, 
                                smooth_sigma=1, 
                                smooth_gradient=1, 
                                delta=1, 
                                n_iter=100, 
                                gradient_decay=0.1,
                                min_area=5, 
                                eps=1e-12, 
                                thresh_factor=None, 
                                mask=None,
                                interp=False,
                                use_connectivity=False, 
                                connectivity_alpha = 0.5,
                                binary_mask_gradient=False,
                                debug_viz=False,
                                renorm_gradient=True,
                                sampling=1000,
                                track_percent=0,
                                rand_seed=0,
                                ref_initial_color_img = None,
                                ref_alpha=0.01,
                                saveplotsfolder=None,
                                viewinit=(0,0)):
    
    """ parses the instance level segmentation implicitly given as an input binary or a vector field. 
    The algorithm works as an inverse watershed.
    
    Step 1: a grid of points is seeds on the image
    Step 2: points are propagated for n_iter according to the gradient_img, condensing towards cell centers implicitly implied by the gradient image.
    Step 3: individual cluster centers are found by binarisation and connected component, removing objects < min_area
    
    result is an integer image the same size as binary. 

    Parameters
    ----------
    binary : (MxNxL) numpy array
        input binary image defining the voxels that need labeling
  	gradient_img :  (MxNxLx3) numpy array
        This is a gradient field such as that from applying np.array(np.gradient(img)).transpose(1,2,3,0) where img is a potential such as a distance transform or probability map. 
    divergence_rescale : 
        If True, the gradient_img is scaled by the divergence which is equivalent to the mean curvature, this helps to prevent early breakage for tube-like structures.   
    smooth_sigma : scalar
        controls the catchment area for identifying distinct cells at the final propagation position. Smaller smooth_sigma leads to more oversegmentation. 
    smooth_gradient : scalar
    	the isotropic sigma value controlling the Gaussian smoothing of the gradient field. More smoothing results in more cells grouped together
    delta: scalar
    	the voxel size to propagate grid points per iteration. Related to the stability. If too small takes too long. If too large, might not converge. if delta=1, takes a 1 voxel step. 
    n_iter: int 
        the number of iterations to run. (To do: monitor convergence and break early to improve speed)
    min_area: scalar
        volume of cells < min_area are removed. 
    eps: float
        a small number for numerical stability
    thresh_factor: scalar
        The final cells are identified by thresholding on a threshold mean+thresh_factor*std. Thresh_factor controls what is an object prior to connected components analysis 
    mask: (MxNxL) numpy array
        optional binary mask to gate the region to parse labels for.
    debug_viz: bool
        if True, visualise the position of the points at every algorithm iteration. 
        
    Returns
    -------
    cell_seg_connected_original : (MxNxL)
        an integer image where each unique int > 0 relates to a unique object such that object 1 is retrieved by cell_seg_connected_original==1.
        
    """
    
    import scipy.ndimage as ndimage
    import numpy as np 
    import skimage.morphology as skmorph
    import pylab as plt 
    import skimage.measure as skmeasure 
    import skimage.segmentation as sksegmentation 
    from tqdm import tqdm 
    from .plotting import set_axes_equal
    
    if gradient_img is not None:
        sdf_normals = gradient_img.transpose(3,0,1,2) # use the supplied gradients! 
        if binary_mask_gradient:
            sdf_normals = sdf_normals * binary[None,...]
    else:
        # compute the signed distance transform
        sdf_normals, sdf_binary = surf_normal_sdf(binary, return_sdf=True, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True)
        if binary_mask_gradient:
            sdf_normals = sdf_normals * binary[None,...]
    
    
    if use_connectivity:
        from sklearn.feature_extraction.image import grid_to_graph
        import scipy.sparse as spsparse 
        
        W = grid_to_graph(binary.shape[0], 
                        binary.shape[1],
                        binary.shape[2], 
                        mask=binary>0)  # convert this to laplacian.   
    
        # create the averaging matrix 
        DD = 1./(W.sum(axis=-1))
        DD = spsparse.spdiags(np.squeeze(DD), [0], DD.shape[0], DD.shape[0]) 
        W = DD.dot(W) # averaging
        del DD # some memory management. 
        
        alpha=connectivity_alpha
        
    if divergence_rescale:
        # rescale the speed
        curvature_3D = mean_curvature_sdf(sdf_normals)
        curvature_3D = _normalize99(curvature_3D, lower=0.01,upper=99) # rescales to a factor between 0-1
        sdf_normals = sdf_normals * curvature_3D[None,...] # multiplicative factor rescaling 
    
    # grid =  np.zeros(binary.shape, dtype=np.int32)
    pts = np.argwhere(binary>0).astype(np.float32) # (N,ndim)
    # pt_ii = np.argwhere(binary>0).astype(np.float32)
    pt_ii = pts.copy()
    
    
    tracks = []
    
    if track_percent > 0:
        np.random.seed(rand_seed)
        n_samples_track = int(track_percent*len(pts))
        # n_samples_track = len(pts)
        select_pts_track = np.arange(len(pts)) #[::1000]
        np.random.shuffle(select_pts_track)
        
        select_pts_track = select_pts_track[:n_samples_track].copy()
        tracks.append(pts[select_pts_track])
        

    mu = momenta # momentum
    # g0 = np.zeros(pts.shape)
    g0 = np.zeros_like(pt_ii)
    
    # delta_position_changes = []
    
    # if use_connectivity:
    #     import point_cloud_utils as pcu 
    #     _, W = pcu.k_nearest_neighbors(pts, pts, k=5)
    #     alpha=connectivity_alpha
    
    
    for ii in tqdm(np.arange(n_iter)):
        # pt_ii = tracks[-1].copy()
        
        """
        interp helps!. 
        """
        if interp:
            pts_vect_ii = np.array([_interp3(pt_ii, binary.shape, I_ref=sdf_normals[ch], method='linear', cast_uint8=False) for ch in np.arange(len(sdf_normals))]).T
        else:
            # print('no interp')
            # pts_vect_ii = sdf_normals[:,
            #                           pt_ii[...,0].astype(np.int32), 
            #                           pt_ii[...,1].astype(np.int32), 
            #                           pt_ii[...,2].astype(np.int32)].T  # direct lookup - not interp!. 
            # faster index with flat indices? 
            pts_vect_ii = (sdf_normals.reshape(3,-1)[:,np.ravel_multi_index(pt_ii.astype(np.int32).T, sdf_normals.shape[1:], mode='raise', order='C')]).T
            
            
        # renormalize
        if renorm_gradient:
            pts_vect_ii = pts_vect_ii / (np.linalg.norm(pts_vect_ii, axis=-1)[:,None] + 1e-20)
            # pts_vect_ii = pts_vect_ii / (np.sqrt(np.sum(pts_vect_ii**2, axis=-1))[:,None] + 1e-20)
            
        """
        Update step:
        """
        if use_connectivity:
            pts_vect_ii = alpha * pts_vect_ii + (1-alpha) * W.dot(pts_vect_ii) # twice
            # pts_vect_ii = alpha * pts_vect_ii + (1-alpha) * np.nanmean(pts_vect_ii[W], axis=1)
            
        # else:
        # pt_ii_next = pt_ii + delta*pts_vect_ii
        pt_ii_next = pt_ii + (delta*pts_vect_ii + mu*g0) * (1./(delta+mu))  * (delta/(1.+ii*gradient_decay))# add momentum + connectivity. 
        
        g0 = pts_vect_ii.copy() # copy this into the past history.... 
        
        pt_ii_next[:,0] = np.clip(pt_ii_next[:,0], 0, binary.shape[0]-1)
        pt_ii_next[:,1] = np.clip(pt_ii_next[:,1], 0, binary.shape[1]-1)
        pt_ii_next[:,2] = np.clip(pt_ii_next[:,2], 0, binary.shape[2]-1)
        
        # delta_change = np.nanmean(np.linalg.norm(pt_ii_next-pt_ii, axis=-1)) # mean or median?  # this can help with early stopping!. 
        # delta_position_changes.append(delta_change)
        # print(delta_change)
        
        # tracks[-1] = pt_ii_next # overwrite 
        pt_ii = pt_ii_next
        
        if track_percent > 0:
            # print(select_pts_track.shape)
            tracks.append(pt_ii_next[select_pts_track,:])
        
        # plt.figure(figsize=(10,10))
        # plt.imshow(binary.max(axis=0))
        # plt.plot(pt_ii_next[:,2], 
        #          pt_ii_next[:,1], 'r.')
        # plt.show(block=False)
        
        if debug_viz:
            
            if ref_initial_color_img is not None:
                ref_pts_color = ref_initial_color_img[pts[::sampling, 0].astype(np.int32),
                                                      pts[::sampling, 1].astype(np.int32),
                                                      pts[::sampling, 2].astype(np.int32)].copy()
                
            
            # sampling = 1000
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_proj_type('ortho') # this works better!.
            ax.set_box_aspect(aspect = (1,1,1)) # this works. 
            # ax.scatter(v_watertight[::sampling,0], 
            #             v_watertight[::sampling,1], 
            #             v_watertight[::sampling,2], 
            #             c='k', s=1, alpha=0.0)#all_labels_branches[np.squeeze(all_dists)<20], s=1)
            
            if ref_initial_color_img is not None:
                ax.scatter(pts[::sampling, 0], 
                           pts[::sampling, 1],
                           pts[::sampling, 2],
                           c =ref_pts_color, 
                           s=1, alpha=ref_alpha, zorder=0) # plot the baseline. 
            else:
                ax.scatter(pts[::sampling, 0], 
                           pts[::sampling, 1],
                           pts[::sampling, 2],
                           c ='k', s=1, alpha=ref_alpha, zorder=0) # plot the baseline. 
            ax.scatter(pt_ii_next[::sampling,0], 
                       pt_ii_next[::sampling,1],
                        pt_ii_next[::sampling,2], c='magenta',s=1, zorder=1000)
                        # pt_ii_next[::sampling,2], c='magenta',s=1)
            
            # ax.scatter(centroids3D_from_xz[:,0], 
            #            centroids3D_from_xz[:,1],
            #            centroids3D_from_xz[:,2], c='g',s=10)
            # ax.scatter(centroids3D_from_yz[:,0], 
            #            centroids3D_from_yz[:,1],
            #            centroids3D_from_yz[:,2], c='b',s=10)
            # # ax.scatter(skel3D_coords[:,0], 
            # #             skel3D_coords[:,1],
            # #             skel3D_coords[:,2], c='k',s=5, alpha=1)
            # ax.view_init(-90,0)
            # ax.view_init(0,180)
            ax.view_init(viewinit[0], viewinit[1])
            # ax.view_init(180,0)
            # ax.set_xlim([0,binary.shape[0]]) # why is this plot not good? 
            # ax.set_ylim([0,binary.shape[1]])
            # ax.set_zlim([0,binary.shape[2]])
            set_axes_equal(ax)
            
            if saveplotsfolder is not None:
                # save the visualization.  
                import os 
                # set this off for better looking visualization. 
                ax.grid('off')
                ax.axis('off')
                plt.savefig(os.path.join(saveplotsfolder, 
                                         'iter_'+str(ii).zfill(8)+'.png'), dpi=120, bbox_inches='tight')
                
            plt.show(block=False)
        
        
# =============================================================================
#     To DO: how can we utilise adaptive stopping? e.g. updating just a fraction of the points. ?     
# =============================================================================
    if len(tracks) > 0:
        tracks = np.array(tracks)
    # # tracks = pt_ii
    
    # # parse ... 
    # votes_grid_acc = np.zeros(binary.shape)
    # # votes_grid_acc[(tracks[-1][:,0]).astype(np.int32), 
    # #                (tracks[-1][:,1]).astype(np.int32),
    # #                (tracks[-1][:,2]).astype(np.int32)] += 1. # add a vote. 
    # votes_grid_acc[(pt_ii[:,0]).astype(np.int32), 
    #                (pt_ii[:,1]).astype(np.int32),
    #                (pt_ii[:,2]).astype(np.int32)] += 1. # add a vote. 
                   
    # # smooth to get a density (fast KDE estimation)
    # votes_grid_acc = ndimage.gaussian_filter(votes_grid_acc, sigma=smooth_sigma)   # gaussian or uniform? 
    
    # if thresh_factor is not None:
    #     if mask is not None:
    #         votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc[mask]) + thresh_factor*np.std(votes_grid_acc[mask])
    #     else:
    #         votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc) + thresh_factor*np.std(votes_grid_acc)
    # else:
    #     votes_grid_binary = votes_grid_acc > np.mean(votes_grid_acc) # just threshold over the mean. 
        
    # cell_seg_connected = skmeasure.label(votes_grid_binary, connectivity=2)
    # cell_uniq_regions = np.setdiff1d(np.unique(cell_seg_connected),0)
    # if len(cell_uniq_regions)>0:
    #     props = skmeasure.regionprops(cell_seg_connected)
    #     areas = np.hstack([re.area for re in props])
    #     invalid_areas = cell_uniq_regions[areas<=min_area]
    
    #     for invalid in invalid_areas:
    #         cell_seg_connected[cell_seg_connected==invalid] = 0
        
    # if cell_seg_connected.max() > 0:
    #     cell_seg_connected = sksegmentation.relabel_sequential(cell_seg_connected)[0]
    
    
    # cell_seg_connected_original = np.zeros_like(cell_seg_connected)
    # # cell_seg_connected_original[(pts[:,0]).astype(np.int32), 
    # #                             (pts[:,1]).astype(np.int32),
    # #                             (pts[:,2]).astype(np.int32)] = cell_seg_connected[(tracks[-1][:,0]).astype(np.int32), 
    # #                                                                               (tracks[-1][:,1]).astype(np.int32),
    # #                                                                               (tracks[-1][:,2]).astype(np.int32)]                                  
    # cell_seg_connected_original[(pts[:,0]).astype(np.int32), 
    #                             (pts[:,1]).astype(np.int32),
    #                             (pts[:,2]).astype(np.int32)] = cell_seg_connected[(pt_ii[:,0]).astype(np.int32), 
    #                                                                               (pt_ii[:,1]).astype(np.int32),
    #                                                                               (pt_ii[:,2]).astype(np.int32)]                                  
    
    # cell_seg_connected_original, cell_seg_connected, votes_grid_acc = connected_components_pts_3D( pts=tracks[-1], 
    #                                                                                                 pts0=pts, 
    #                                                                                                 shape=binary.shape[:3], 
    #                                                                                                 smooth_sigma=smooth_sigma, 
    #                                                                                                 thresh_factor=thresh_factor, 
    #                                                                                                 mask=mask,
    #                                                                                                 min_area=min_area)
    cell_seg_connected_original, cell_seg_connected, votes_grid_acc = connected_components_pts_3D( pts=pt_ii, 
                                                                                                    pts0=pts, 
                                                                                                    shape=binary.shape[:3], 
                                                                                                    smooth_sigma=smooth_sigma, 
                                                                                                    thresh_factor=thresh_factor, 
                                                                                                    mask=mask,
                                                                                                    min_area=min_area)
    
    # return cell_seg_connected_original, cell_seg_connected, tracks, votes_grid_acc
    return cell_seg_connected_original, cell_seg_connected, tracks, votes_grid_acc



# isolating out just the dynamics
# can we make this v. fast? ---> it is mainly the advection that is slow....., if we could parallelize in blocks? 
def gradient_watershed3D_binary_dynamics(binary, 
                                    gradient_img=None, 
                                    momenta = 0.75,
                                    divergence_rescale=False, 
                                    smooth_sigma=1, 
                                    smooth_gradient=1, 
                                    gradient_decay=0.0,
                                    delta=1, 
                                    n_iter=100, 
                                    min_area=5, 
                                    eps=1e-12, 
                                    thresh_factor=None, 
                                    mask=None,
                                    interp=False,
                                    use_connectivity=False, 
                                    connectivity_alpha = 0.5,
                                    debug_viz=False,
                                    binary_mask_gradient = False,
                                    renorm_gradient=True):
    
    """ parses the instance level segmentation implicitly given as an input binary or a vector field. 
    The algorithm works as an inverse watershed.
    
    Step 1: a grid of points is seeds on the image
    Step 2: points are propagated for n_iter according to the gradient_img, condensing towards cell centers implicitly implied by the gradient image.
    Step 3: individual cluster centers are found by binarisation and connected component, removing objects < min_area
    
    result is an integer image the same size as binary. 

    Parameters
    ----------
    binary : (MxNxL) numpy array
        input binary image defining the voxels that need labeling
  	gradient_img :  (MxNxLx3) numpy array
        This is a gradient field such as that from applying np.array(np.gradient(img)).transpose(1,2,3,0) where img is a potential such as a distance transform or probability map. 
    divergence_rescale : 
        If True, the gradient_img is scaled by the divergence which is equivalent to the mean curvature, this helps to prevent early breakage for tube-like structures.   
    smooth_sigma : scalar
        controls the catchment area for identifying distinct cells at the final propagation position. Smaller smooth_sigma leads to more oversegmentation. 
    smooth_gradient : scalar
    	the isotropic sigma value controlling the Gaussian smoothing of the gradient field. More smoothing results in more cells grouped together
    delta: scalar
    	the voxel size to propagate grid points per iteration. Related to the stability. If too small takes too long. If too large, might not converge. if delta=1, takes a 1 voxel step. 
    n_iter: int 
        the number of iterations to run. (To do: monitor convergence and break early to improve speed)
    min_area: scalar
        volume of cells < min_area are removed. 
    eps: float
        a small number for numerical stability
    thresh_factor: scalar
        The final cells are identified by thresholding on a threshold mean+thresh_factor*std. Thresh_factor controls what is an object prior to connected components analysis 
    mask: (MxNxL) numpy array
        optional binary mask to gate the region to parse labels for.
    debug_viz: bool
        if True, visualise the position of the points at every algorithm iteration. 
        
    Returns
    -------
    cell_seg_connected_original : (MxNxL)
        an integer image where each unique int > 0 relates to a unique object such that object 1 is retrieved by cell_seg_connected_original==1.
        
    """
    
    import scipy.ndimage as ndimage
    import numpy as np 
    import skimage.morphology as skmorph
    import pylab as plt 
    import skimage.measure as skmeasure 
    import skimage.segmentation as sksegmentation 
    from tqdm import tqdm 
    from .plotting import set_axes_equal
    
    if gradient_img is not None:
        sdf_normals = gradient_img.transpose(3,0,1,2) # use the supplied gradients! 
        if binary_mask_gradient:
            sdf_normals = sdf_normals * binary[None,...]
    else:
        # compute the signed distance transform
        sdf_normals, sdf_binary = surf_normal_sdf(binary, return_sdf=True, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True)
        if binary_mask_gradient:
            sdf_normals = sdf_normals * binary[None,...]
    
    
    if use_connectivity:
        from sklearn.feature_extraction.image import grid_to_graph
        import scipy.sparse as spsparse 
        
        W = grid_to_graph(binary.shape[0], 
                        binary.shape[1],
                        binary.shape[2], 
                        mask=binary>0)  # convert this to laplacian.   
    
        # create the averaging matrix 
        DD = 1./(W.sum(axis=-1))
        DD = spsparse.spdiags(np.squeeze(DD), [0], DD.shape[0], DD.shape[0]) 
        W = DD.dot(W) # averaging
        del DD # some memory management. 
        
        alpha=connectivity_alpha
        
    if divergence_rescale:
        # rescale the speed
        curvature_3D = mean_curvature_sdf(sdf_normals)
        curvature_3D = _normalize99(curvature_3D, lower=0.01,upper=99) # rescales to a factor between 0-1
        sdf_normals = sdf_normals * curvature_3D[None,...] # multiplicative factor rescaling 
    
    # grid =  np.zeros(binary.shape, dtype=np.int32)
    pts = np.argwhere(binary>0).astype(np.float32) # (N,ndim)
    # pt_ii = np.argwhere(binary>0).astype(np.float32)
    pt_ii = pts.copy()
    
    # tracks = [pts]
    mu = momenta # momentum
    # g0 = np.zeros(pts.shape)
    g0 = np.zeros_like(pt_ii)
    
    
    for ii in np.arange(n_iter):
        
        """
        interp helps!. 
        """
        if interp:
            pts_vect_ii = np.array([_interp3(pt_ii, binary.shape, I_ref=sdf_normals[ch], method='linear', cast_uint8=False) for ch in np.arange(len(sdf_normals))]).T
        else:
            # print('no interp')
            # pts_vect_ii = sdf_normals[:,
            #                           pt_ii[...,0].astype(np.int32), 
            #                           pt_ii[...,1].astype(np.int32), 
            #                           pt_ii[...,2].astype(np.int32)].T  # direct lookup - not interp!. 
            # faster index with flat indices? 
            pts_vect_ii = (sdf_normals.reshape(3,-1)[:,np.ravel_multi_index(pt_ii.astype(np.int32).T, sdf_normals.shape[1:], mode='raise', order='C')]).T
            
            
        # renormalize
        if renorm_gradient:
            pts_vect_ii = pts_vect_ii / (np.linalg.norm(pts_vect_ii, axis=-1)[:,None] + 1e-20)
        
        
        """
        Update step:
        """
        if use_connectivity:
            pts_vect_ii = alpha * pts_vect_ii + (1-alpha) * W.dot(pts_vect_ii) # twice
            # pts_vect_ii = alpha * pts_vect_ii + (1-alpha) * np.nanmean(pts_vect_ii[W], axis=1)
            
        # else:
        # pt_ii_next = pt_ii + delta*pts_vect_ii
        # pt_ii_next = pt_ii + (delta*pts_vect_ii + mu*g0) * (1./(delta+mu))  # add momentum + connectivity. 
        pt_ii_next = pt_ii + (delta*pts_vect_ii + mu*g0) * (1./(delta+mu))  * (delta/(1.+ii*gradient_decay))# add momentum + connectivity. 
        g0 = pts_vect_ii.copy() # copy this into the past history.... 
        
        pt_ii_next[:,0] = np.clip(pt_ii_next[:,0], 0, binary.shape[0]-1)
        pt_ii_next[:,1] = np.clip(pt_ii_next[:,1], 0, binary.shape[1]-1)
        pt_ii_next[:,2] = np.clip(pt_ii_next[:,2], 0, binary.shape[2]-1)
        
        # delta_change = np.nanmean(np.linalg.norm(pt_ii_next-pt_ii, axis=-1)) # mean or median?  # this can help with early stopping!. 
        # delta_position_changes.append(delta_change)
        # print(delta_change)
        
        # tracks[-1] = pt_ii_next # overwrite 
        pt_ii = pt_ii_next
        
        # plt.figure(figsize=(10,10))
        # plt.imshow(binary.max(axis=0))
        # plt.plot(pt_ii_next[:,2], 
        #          pt_ii_next[:,1], 'r.')
        # plt.show(block=False)
        
        if debug_viz:
            sampling = 1000
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_proj_type('ortho') # this works better!.
            ax.set_box_aspect(aspect = (1,1,1)) # this works. 
            # ax.scatter(v_watertight[::sampling,0], 
            #             v_watertight[::sampling,1], 
            #             v_watertight[::sampling,2], 
            #             c='k', s=1, alpha=0.0)#all_labels_branches[np.squeeze(all_dists)<20], s=1)
            ax.scatter(pt_ii_next[::sampling,0], 
                       pt_ii_next[::sampling,1],
                       pt_ii_next[::sampling,2], c='r',s=1)
            # ax.scatter(centroids3D_from_xz[:,0], 
            #            centroids3D_from_xz[:,1],
            #            centroids3D_from_xz[:,2], c='g',s=10)
            # ax.scatter(centroids3D_from_yz[:,0], 
            #            centroids3D_from_yz[:,1],
            #            centroids3D_from_yz[:,2], c='b',s=10)
            # # ax.scatter(skel3D_coords[:,0], 
            # #             skel3D_coords[:,1],
            # #             skel3D_coords[:,2], c='k',s=5, alpha=1)
            # ax.view_init(-90,0)
            ax.view_init(0,180)
            # ax.view_init(180,0)
            # ax.set_xlim([0,binary.shape[0]]) # why is this plot not good? 
            # ax.set_ylim([0,binary.shape[1]])
            # ax.set_zlim([0,binary.shape[2]])
            set_axes_equal(ax)
            plt.show(block=False)
            
    # print(ii)
    return np.concatenate([pts[None,...], pt_ii[None,...]], axis=0) # return the initial and final position!. 


def mp_gradient_watershed3D_binary(binary, 
                                   gradient_img=None, 
                                   tile_shape=None,
                                   tile_aspect=(1,2,2),
                                   tile_overlap_ratio = 0.25, # should be bigger than the most movement. 
                                   momenta = 0.95,
                                   divergence_rescale=False, 
                                   gradient_decay=0.0,
                                   smooth_sigma=1, 
                                   smooth_gradient=1, 
                                   delta=1, 
                                   n_iter=100, 
                                   min_area=5, 
                                   eps=1e-12, 
                                   thresh_factor=None, 
                                   mask=None,
                                   interp=False,
                                   use_connectivity=False, 
                                   connectivity_alpha = 0.5,
                                   binary_mask_gradient = False, 
                                   debug_viz=False,
                                   renorm_gradient=False):

    import multiprocess as mp 
    from multiprocess.pool import ThreadPool
    import numpy as np 
    from tiler import Tiler
    
    n_procs = mp.cpu_count() - 1
    
    if tile_shape is None:
        # do autoinfer. 
        total_vol = np.prod(binary.shape[:3])
        # per_vol = total_vol / (n_procs /(1.+tile_overlap_ratio))
        per_vol = total_vol / n_procs #/2. # assume we have less!. 
        
        # not sure the division is correct.
        tile_shape = tuple(np.int32(np.hstack( (per_vol / (float(np.sum(tile_aspect))))**(1./3) * np.hstack(tile_aspect))))
        # tile_shape = tuple(np.int32(np.hstack((total_vol/(n_procs/(1.)))/ (float(np.sum(tile_aspect))) * np.hstack(tile_aspect))))
        
        smallest_ind = np.argmin(binary.shape)
        max_correction_factor = np.int32(np.ceil(binary.shape[smallest_ind] / tile_shape[smallest_ind]))
        
        overlap = np.int32(np.hstack(tile_shape)*tile_overlap_ratio)
        
        tiler = Tiler(
                    data_shape=binary.shape,
                    tile_shape=tile_shape,
                    overlap=overlap #, # maintain approx 10%
                    )
        correction_factor = (tiler.n_tiles*1./tile_overlap_ratio/n_procs)**(1./3)
        # print(correction_factor)
        correction_factor = np.minimum( correction_factor, max_correction_factor)
        tile_shape = tuple(np.int32(np.hstack(tile_shape) * correction_factor))
        
        print('using tile shape: ', tile_shape)
        
    overlap = np.int32(np.hstack(tile_shape)*tile_overlap_ratio)
    
    tiler = Tiler(
                data_shape=binary.shape,
                tile_shape=tile_shape,
                overlap=overlap#, # maintain approx 10%
                )

    # the total number will change after this!.     
    # now we need to create the volume with extra padding i guess. 
    # Calculate and apply extra padding, as well as adjust tiling parameters
    new_shape, padding = tiler.calculate_padding()
    print(new_shape, padding)
    

    offset = np.hstack([p[0] for p in padding])
    tiler.recalculate(data_shape=new_shape)
    # padded_volume = np.pad(volume, padding, mode="reflect")
    
    shape = binary.shape
    
    if gradient_img is not None:
        sdf_normals = gradient_img.transpose(3,0,1,2) # use the supplied gradients! 
        if binary_mask_gradient:
            sdf_normals = sdf_normals * binary[None,...]
    else:
        # compute the signed distance transform
        sdf_normals, sdf_binary = surf_normal_sdf(binary, return_sdf=True, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True)
        if binary_mask_gradient:
            sdf_normals = sdf_normals * binary[None,...]
    
    im_binary = np.pad(binary, padding, mode='constant')
    sdf_normals = np.pad(sdf_normals, [(0,0)]+padding)
    
    print('using total number of tiles : ', tiler.n_tiles) #### why are there more tiles? ok.... so the extra padding vastly created more? why ? 
    
    """
    Wrap the basic function to multiprocess each tile.
    """
    def _run_dynamics(tile_id):
        
        bbox_low, bbox_high = tiler.get_tile_bbox(tile_id)
        size = np.hstack(bbox_high) - np.hstack(bbox_low)
        res = gradient_watershed3D_binary_dynamics(im_binary[bbox_low[0]:bbox_high[0],bbox_low[1]:bbox_high[1], bbox_low[2]:bbox_high[2]], 
                                                            gradient_img = sdf_normals[:,bbox_low[0]:bbox_high[0],bbox_low[1]:bbox_high[1],bbox_low[2]:bbox_high[2]].transpose(1,2,3,0),
                                                            momenta=momenta, 
                                                            divergence_rescale=divergence_rescale, # i don't think this was defined correct helps.... ----> this actually seems to help by putting random fiducials...... (not actually stable..) 
                                                            smooth_sigma=smooth_sigma, # use a larger for weird shapes!.  
                                                            smooth_gradient=smooth_gradient, # this has no effect except divergence_rescale=True
                                                            gradient_decay = gradient_decay,
                                                            delta=delta, # kept at 1 for 1px hop. 
                                                            n_iter=n_iter, # evolve less - the limit isn't very stable 
                                                            eps=eps, 
                                                            min_area=min_area,
                                                            thresh_factor=None,
                                                            # mask=im_binary,
                                                            use_connectivity=use_connectivity,
                                                            connectivity_alpha=connectivity_alpha,
                                                            renorm_gradient=renorm_gradient,
                                                            binary_mask_gradient=binary_mask_gradient,
                                                            debug_viz=False) # 50 seems not enough for 3D... as it is very connected. 

        if len(res)>0:
            # # keep only the points starting within the non-overlapped region!. 
            # # trunk out the overlapping region 
            keep = np.ones(res[0].shape[:1], dtype=bool)
            # also remove those that didn't move a lot
            # keep = np.linalg.norm(res[0]-res[1], axis=-1) > 2 # no point keeping fixed points. # they should all move
            print('Tile id', tile_id, len(keep), np.sum(keep))
            
            for dim in np.arange(res.shape[-1]):
                keep = np.logical_and(keep, 
                                      np.logical_and( res[0,...,dim] >= overlap[dim]//2, 
                                                      res[0,...,dim] < size[dim]-overlap[dim]//2)) # only keep 
            
            # # use broadcasting
            # keep = np.logical_and( res[0]>= (overlap//2)[None,:], res[0] < (size - overlap//2)[None,:])
            
            res = res[:,keep>0] + np.hstack(bbox_low)[None,None,:] # then add back the coordinate axis. 
        return res
    
    print('start dynamics')
    import time 
    t1 = time.time()
    
    # with mp.Pool(n_procs) as pool:
    with ThreadPool(n_procs) as pool: 
        res = pool.map(_run_dynamics, range(0, tiler.n_tiles)) # why is this runnin more than expected? 
        res = np.concatenate(res, 1) # collapse all. 
        # res = np.concatenate([result.get() for result in res], axis=1)
        res = res - offset[None,None,:] # subtract the global padding.... everything else shud be fine!. 
    
    t2 = time.time()
    print('finished dynamics:, ', t2-t1)
    
    # remove the offset to get the correct. 
    im_binary = im_binary[padding[0][0]:padding[0][0]+shape[0], 
                          padding[1][0]:padding[1][0]+shape[1], 
                          padding[2][0]:padding[2][0]+shape[2]]

    # we need to clip the points to the original shape. 
    res[...,0] = np.clip(res[...,0], 0, shape[0]-1)
    res[...,1] = np.clip(res[...,1], 0, shape[1]-1)
    res[...,2] = np.clip(res[...,2], 0, shape[2]-1)

    cell_seg_connected_original, cell_seg_connected, votes_grid_acc = connected_components_pts_3D( pts=res[1], 
                                                                                                    pts0=res[0], 
                                                                                                    shape=shape, 
                                                                                                    smooth_sigma=smooth_sigma, 
                                                                                                    thresh_factor=thresh_factor, 
                                                                                                    mask=im_binary,
                                                                                                    min_area=min_area)

    return cell_seg_connected_original, cell_seg_connected, res[1], votes_grid_acc

# def gradient_watershed3D_binary_mp(binary, 
#                                 gradient_img=None, 
#                                 momenta = 0.75,
#                                 divergence_rescale=True, 
#                                 smooth_sigma=1, 
#                                 smooth_gradient=1, 
#                                 delta=1, 
#                                 n_iter=100, 
#                                 min_area=5, 
#                                 eps=1e-12, 
#                                 thresh_factor=None, 
#                                 mask=None,
#                                 interp=False,
#                                 use_connectivity=False, 
#                                 connectivity_alpha = 0.5,
#                                 debug_viz=False,
#                                 renorm_gradient=True):
    
#     """ parses the instance level segmentation implicitly given as an input binary or a vector field. 
#     The algorithm works as an inverse watershed.
    
#     Step 1: a grid of points is seeds on the image
#     Step 2: points are propagated for n_iter according to the gradient_img, condensing towards cell centers implicitly implied by the gradient image.
#     Step 3: individual cluster centers are found by binarisation and connected component, removing objects < min_area
    
#     result is an integer image the same size as binary. 

#     Parameters
#     ----------
#     binary : (MxNxL) numpy array
#         input binary image defining the voxels that need labeling
#   	gradient_img :  (MxNxLx3) numpy array
#         This is a gradient field such as that from applying np.array(np.gradient(img)).transpose(1,2,3,0) where img is a potential such as a distance transform or probability map. 
#     divergence_rescale : 
#         If True, the gradient_img is scaled by the divergence which is equivalent to the mean curvature, this helps to prevent early breakage for tube-like structures.   
#     smooth_sigma : scalar
#         controls the catchment area for identifying distinct cells at the final propagation position. Smaller smooth_sigma leads to more oversegmentation. 
#     smooth_gradient : scalar
#     	the isotropic sigma value controlling the Gaussian smoothing of the gradient field. More smoothing results in more cells grouped together
#     delta: scalar
#     	the voxel size to propagate grid points per iteration. Related to the stability. If too small takes too long. If too large, might not converge. if delta=1, takes a 1 voxel step. 
#     n_iter: int 
#         the number of iterations to run. (To do: monitor convergence and break early to improve speed)
#     min_area: scalar
#         volume of cells < min_area are removed. 
#     eps: float
#         a small number for numerical stability
#     thresh_factor: scalar
#         The final cells are identified by thresholding on a threshold mean+thresh_factor*std. Thresh_factor controls what is an object prior to connected components analysis 
#     mask: (MxNxL) numpy array
#         optional binary mask to gate the region to parse labels for.
#     debug_viz: bool
#         if True, visualise the position of the points at every algorithm iteration. 
        
#     Returns
#     -------
#     cell_seg_connected_original : (MxNxL)
#         an integer image where each unique int > 0 relates to a unique object such that object 1 is retrieved by cell_seg_connected_original==1.
        
#     """
    
#     import scipy.ndimage as ndimage
#     import numpy as np 
#     import skimage.morphology as skmorph
#     import pylab as plt 
#     import skimage.measure as skmeasure 
#     import skimage.segmentation as sksegmentation 
#     from tqdm import tqdm 
#     from .plotting import set_axes_equal
#     import multiprocess as mp
    
#     n_cpu = mp.cpu_count() - 1
    
    
#     if gradient_img is not None:
#         sdf_normals = gradient_img.transpose(3,0,1,2) # use the supplied gradients! 
#         sdf_normals = sdf_normals * binary[None,...]
#     else:
#         # compute the signed distance transform
#         sdf_normals, sdf_binary = surf_normal_sdf(binary, return_sdf=True, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True)
#         sdf_normals = sdf_normals * binary[None,...]
    
    
#     if use_connectivity:
#         from sklearn.feature_extraction.image import grid_to_graph
#         import scipy.sparse as spsparse 
        
#         W = grid_to_graph(binary.shape[0], 
#                         binary.shape[1],
#                         binary.shape[2], 
#                         mask=binary>0)  # convert this to laplacian.   
    
#         # create the averaging matrix 
#         DD = 1./(W.sum(axis=-1))
#         DD = spsparse.spdiags(np.squeeze(DD), [0], DD.shape[0], DD.shape[0]) 
#         W = DD.dot(W) # averaging
#         del DD # some memory management. 
        
#         alpha=connectivity_alpha
        
#     if divergence_rescale:
#         # rescale the speed
#         curvature_3D = mean_curvature_sdf(sdf_normals)
#         curvature_3D = _normalize99(curvature_3D, lower=0.01,upper=99) # rescales to a factor between 0-1
#         sdf_normals = sdf_normals * curvature_3D[None,...] # multiplicative factor rescaling 
    
#     # grid =  np.zeros(binary.shape, dtype=np.int32)
#     pts = np.argwhere(binary>0).astype(np.float32) # (N,ndim)
#     # pt_ii = np.argwhere(binary>0).astype(np.float32)
#     pt_ii = pts.copy()
    
#     # tracks = [pts]
#     mu = momenta # momentum
#     # g0 = np.zeros(pts.shape)
#     g0 = np.zeros_like(pt_ii)
    
#     # delta_position_changes = []
    
#     # if use_connectivity:
#     #     import point_cloud_utils as pcu 
#     #     _, W = pcu.k_nearest_neighbors(pts, pts, k=5)
#     #     alpha=connectivity_alpha
    
    
#     for ii in tqdm(np.arange(n_iter)):
#         # pt_ii = tracks[-1].copy()
        
#         """
#         interp helps!. 
#         """
#         if interp:
#             pts_vect_ii = np.array([_interp3(pt_ii, binary.shape, I_ref=sdf_normals[ch], method='linear', cast_uint8=False) for ch in np.arange(len(sdf_normals))]).T
#         else:
#             # print('no interp')
#             # pts_vect_ii = sdf_normals[:,
#             #                           pt_ii[...,0].astype(np.int32), 
#             #                           pt_ii[...,1].astype(np.int32), 
#             #                           pt_ii[...,2].astype(np.int32)].T  # direct lookup - not interp!. 
#             # faster index with flat indices? 
#             pts_vect_ii = (sdf_normals.reshape(3,-1)[:,np.ravel_multi_index(pt_ii.astype(np.int32).T, sdf_normals.shape[1:], mode='raise', order='C')]).T
            
            
#         # renormalize
#         if renorm_gradient:
#             pts_vect_ii = pts_vect_ii / (np.linalg.norm(pts_vect_ii, axis=-1)[:,None] + 1e-20)
#             # pts_vect_ii = pts_vect_ii / (np.sqrt(np.sum(pts_vect_ii**2, axis=-1))[:,None] + 1e-20)
            
#         """
#         Update step:
#         """
#         if use_connectivity:
#             pts_vect_ii = alpha * pts_vect_ii + (1-alpha) * W.dot(pts_vect_ii) # twice
#             # pts_vect_ii = alpha * pts_vect_ii + (1-alpha) * np.nanmean(pts_vect_ii[W], axis=1)
            
#         # # else:
#         # # pt_ii_next = pt_ii + delta*pts_vect_ii
#         pt_ii_next = pt_ii + (delta*pts_vect_ii + mu*g0) * (1./(delta+mu))  # add momentum + connectivity. 
        
#         # pt_ii_ids = np.array_split(np.arange(len(pt_ii)), n_cpu)
        
#         # def mp_sum(ids):
#         #     pp = pt_ii[ids] + (delta*pts_vect_ii[ids] + mu*g0[ids]) * (1./(delta+mu)) 
#         #     return pp
        
#         # with mp.Pool() as pool:
#         #     pt_ii_next = pool.map(mp_sum, pt_ii_ids)
#         #     pt_ii_next = np.vstack(pt_ii_next)
        
#         g0 = pts_vect_ii.copy() # copy this into the past history.... 
        
#         pt_ii_next[:,0] = np.clip(pt_ii_next[:,0], 0, binary.shape[0]-1)
#         pt_ii_next[:,1] = np.clip(pt_ii_next[:,1], 0, binary.shape[1]-1)
#         pt_ii_next[:,2] = np.clip(pt_ii_next[:,2], 0, binary.shape[2]-1)
        
#         # delta_change = np.nanmean(np.linalg.norm(pt_ii_next-pt_ii, axis=-1)) # mean or median?  # this can help with early stopping!. 
#         # delta_position_changes.append(delta_change)
#         # print(delta_change)
        
#         # tracks[-1] = pt_ii_next # overwrite 
#         pt_ii = pt_ii_next
        
#         # plt.figure(figsize=(10,10))
#         # plt.imshow(binary.max(axis=0))
#         # plt.plot(pt_ii_next[:,2], 
#         #          pt_ii_next[:,1], 'r.')
#         # plt.show(block=False)
        
#         if debug_viz:
#             sampling = 1000
#             fig = plt.figure(figsize=(10,10))
#             ax = fig.add_subplot(111, projection='3d')
#             ax.set_proj_type('ortho') # this works better!.
#             ax.set_box_aspect(aspect = (1,1,1)) # this works. 
#             # ax.scatter(v_watertight[::sampling,0], 
#             #             v_watertight[::sampling,1], 
#             #             v_watertight[::sampling,2], 
#             #             c='k', s=1, alpha=0.0)#all_labels_branches[np.squeeze(all_dists)<20], s=1)
#             ax.scatter(pt_ii_next[::sampling,0], 
#                        pt_ii_next[::sampling,1],
#                        pt_ii_next[::sampling,2], c='r',s=1)
#             # ax.scatter(centroids3D_from_xz[:,0], 
#             #            centroids3D_from_xz[:,1],
#             #            centroids3D_from_xz[:,2], c='g',s=10)
#             # ax.scatter(centroids3D_from_yz[:,0], 
#             #            centroids3D_from_yz[:,1],
#             #            centroids3D_from_yz[:,2], c='b',s=10)
#             # # ax.scatter(skel3D_coords[:,0], 
#             # #             skel3D_coords[:,1],
#             # #             skel3D_coords[:,2], c='k',s=5, alpha=1)
#             # ax.view_init(-90,0)
#             ax.view_init(0,180)
#             # ax.view_init(180,0)
#             # ax.set_xlim([0,binary.shape[0]]) # why is this plot not good? 
#             # ax.set_ylim([0,binary.shape[1]])
#             # ax.set_zlim([0,binary.shape[2]])
#             set_axes_equal(ax)
#             plt.show(block=False)
        
        
# # =============================================================================
# #     To DO: how can we utilise adaptive stopping? e.g. updating just a fraction of the points. ?     
# # =============================================================================
#     cell_seg_connected_original, cell_seg_connected, votes_grid_acc = connected_components_pts_3D( pts=pt_ii, 
#                                                                                                     pts0=pts, 
#                                                                                                     shape=binary.shape[:3], 
#                                                                                                     smooth_sigma=smooth_sigma, 
#                                                                                                     thresh_factor=thresh_factor, 
#                                                                                                     mask=mask,
#                                                                                                     min_area=min_area)
    
#     # return cell_seg_connected_original, cell_seg_connected, tracks, votes_grid_acc
#     return cell_seg_connected_original, cell_seg_connected, pt_ii, votes_grid_acc








