# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:35:23 2023

@author: fyz11
"""
from segment3D import filters as usegment3D_filters
from segment3D import gpu as usegment3D_gpu
from segment3D import segmentation as usegment3D_segment
from segment3D import watershed as usegment3D_watershed
from segment3D import flows as usegment3D_flows
from segment3D import file_io as usegment3D_fio

import scipy.ndimage as ndimage 
import numpy as np 
import os 


def preprocess_imgs(img, params):
    r""" wrapper function for running preprocessing steps. 

    Parameters
    ----------
    img : numpy array
        (M,N,L) grayscale 3D volume image or (C,M,N,L) multichannel 3D volumes where C=# of channels. preprocessing will be applied to each channel separately.
    params : Python dict
        Dictionary of parameters as specified by segment3D.parameters.get_preprocess_params

    Returns
    -------
    imgs_out : numpy array
        (M,N,L) grayscale 3D volume image or (C,M,N,L) multichannel 3D volumes where C=# of channels. preprocessing will be applied to each channel separately. If do_img_avg == True in the parameters, channels will be averaged to return a single (M,N,L) volume image
    
    """
    
    # check dimensions
    img_dim = len(img.shape)
    
    if img_dim > 3:
        imgs_out = img.copy()
    elif img_dim == 3:
        imgs_out = [img]
    else:
        print('error: Image must be a grayscale volume image of 3 spatial dimensions, or be 4D array where the first dimension, is the number of images or channels.')
        
    
    # 1. resize and fill in gaps with median 
    zoom = [params['factor'] * p for p in params['voxel_res']]
    
    try:
        imgs_out = np.array([usegment3D_gpu.dask_cuda_rescale(usegment3D_filters.fill_array(imgs_out[ch], method='median'), zoom, order=1, mode='reflect') for ch in np.arange(len(imgs_out))], dtype=np.float32)
    except:
        try:
            print('no CUDA. trying torch for resizing')
            imgs_out = np.array([usegment3D_gpu.zoom_3d_pytorch(usegment3D_filters.fill_array(imgs_out[ch], method='median'), zoom) for ch in np.arange(len(imgs_out))], dtype=np.float32)
        except:
            print('no gpu. falling back to CPU for resizing')
            # imgs_out = np.array([ndimage.zoom(usegment3D_filters.fill_array(imgs_out[ch], method='median'), zoom, order=1, mode='reflect') for ch in np.arange(len(imgs_out))], dtype=np.float32)
            imgs_out = np.array([usegment3D_gpu.dask_cpu_rescale(usegment3D_filters.fill_array(imgs_out[ch], method='median'), zoom, order=1, mode='reflect') for ch in np.arange(len(imgs_out))], dtype=np.float32)

    # 2. background correct illumination 
    bg_ds = params['bg_ds']
    bg_sigma = params['bg_sigma']
    bg_normalize_min = params['normalize_min']
    bg_normalize_max = params['normalize_max']
    do_bg_correct = params['do_bg_correction']
    
    if do_bg_correct:
        try:
            # gpu.bg_normalize uses cucim which somehow needs module load cuda. # so does cupy give this error hm... 
            imgs_out = np.array([usegment3D_gpu.num_normalize(usegment3D_gpu.bg_normalize(ch_im, bg_ds=bg_ds, bg_sigma=bg_sigma), pmin=bg_normalize_min, pmax=bg_normalize_max, clip=True) for ch_im in imgs_out])
            # imgs_out = np.array([usegment3D_gpu.num_normalize(usegment3D_gpu.dask_cuda_bg(ch_im, 
            #                                                                               bg_ds=bg_ds, 
            #                                                                               bg_sigma=bg_sigma), pmin=bg_normalize_min, pmax=bg_normalize_max, clip=True) for ch_im in imgs_out])
        except:
            try:
                print('no CUDA. trying torch for normalizing') 
                imgs_out = np.array([usegment3D_gpu.num_normalize(usegment3D_gpu.bg_normalize_torch(ch_im, bg_ds=bg_ds, bg_sigma=bg_sigma), pmin=bg_normalize_min, pmax=bg_normalize_max, clip=True) for ch_im in imgs_out])
            except:
                print('no gpu. falling back to CPU for normalizing')
                imgs_out = np.array([usegment3D_gpu.num_normalize(usegment3D_gpu.bg_normalize_cpu(ch_im, bg_ds=bg_ds, bg_sigma=bg_sigma), pmin=bg_normalize_min, pmax=bg_normalize_max, clip=True) for ch_im in imgs_out])
    else:
        try:
            # gpu.bg_normalize uses cucim which somehow needs module load cuda. # so does cupy give this error hm... 
            imgs_out = np.array([usegment3D_gpu.num_normalize(ch_im, pmin=bg_normalize_min, pmax=bg_normalize_max, clip=True) for ch_im in imgs_out])
            # imgs_out = np.array([usegment3D_gpu.num_normalize(usegment3D_gpu.dask_cuda_bg(ch_im, 
            #                                                                               bg_ds=bg_ds, 
            #                                                                               bg_sigma=bg_sigma), pmin=bg_normalize_min, pmax=bg_normalize_max, clip=True) for ch_im in imgs_out])
        except:
            print('no gpu. falling back to CPU for normalizing')
            imgs_out = np.array([usegment3D_gpu.num_normalize(ch_im, pmin=bg_normalize_min, pmax=bg_normalize_max, clip=True) for ch_im in imgs_out])
   
        
    # 3. combine channels if needed.
    if params['do_avg_imgs']: 
        avg_func = params['avg_func_imgs']
        imgs_out = avg_func(imgs_out, axis=0)
        
    return imgs_out
        
        
def Cellpose2D_model_auto(img, view, params, basename=None, savefolder=None):
    
    r""" wrapper function for running Cellpose2D with diameter autotuning. 

    Parameters
    ----------
    img : numpy array
        (M,N,L) grayscale 3D volume image or (C,M,N,L) multichannel 3D volumes where C=# of channels. This image should be provided in the same order as the input i.e. NO TRANSPOSE needed
    view : one of 'xy', 'xz', or 'yz'
        specifies the view. 
    params : Python dict
        Dictionary of parameters as specified by segment3D.parameters.get_preprocess_params

    Returns
    -------
    auto_diam : Python dict
        (diam_range, diam_score, best_diam, test_slice
    all_probs : numpy array 
    
    all_flows : numpy array 
    
    all_styles : numpy array 


    """
    from cellpose import models
    import scipy.io as spio 
    import cellpose
    
    if int(cellpose.version.split('.')[0])<4:
        model = models.Cellpose(model_type=params['cellpose_modelname'], gpu=params['gpu'])
    else:
        model = models.CellposeModel(pretrained_model=params['cellpose_modelname'], gpu=params['gpu'])
    
    grayscale_bool = params['cellpose_channels'] =='grayscale'
    if grayscale_bool:
        channels = [0,0]
    else:
        channels = [2,1] # cytoplasm = 'Green' and nuclei='Red' channel.
    
    
    if view == 'xy':
        order = (0,1,2,3)
    elif view == 'xz':
        order = (1,0,2,3)
    elif view == 'yz':
        order = (2,0,1,3)
    else:
        print('error: view parameter must be one of the strings, \'xy\', \'xz\', or \'yz\' ') 
        
    if grayscale_bool:
        input_im = img.transpose(order)[...,0]
    else:
        input_im = img.transpose(order)
    
    auto_diam, (all_probs, all_flows, all_styles) = usegment3D_segment.apply_cellpose_model_2D_prob(input_im, 
                                                                                                    model, 
                                                                                                    model_channels=channels, 
                                                                                                    best_diam=params['best_diam'], 
                                                                                                    use_Cellpose_auto_diameter=params['use_Cellpose_auto_diameter'],
                                                                                                    model_invert=params['model_invert'], 
                                                                                                    test_slice=params['test_slice'], 
                                                                                                    hist_norm=params['hist_norm'], # was True
                                                                                                    diam_range=params['diam_range'], 
                                                                                                    ksize=params['ksize'],
                                                                                                    smoothwinsize=params['smoothwinsize'],
                                                                                                    kernel_size=params['histnorm_kernel_size'], 
                                                                                                    clip_limit=params['histnorm_clip_limit'],
                                                                                                    use_edge=params['use_edge'],
                                                                                                    show_img_bounds = params['show_img_bounds'],
                                                                                                    saveplotsfolder=params['saveplotsfolder'],
                                                                                                    use_prob_weighted_score=params['use_prob_weighted_score'],
                                                                                                    debug_viz=params['debug_viz']) 

    all_probs = np.squeeze(all_probs).astype(np.float32)
    all_flows = np.squeeze(all_flows).astype(np.float32) 
            
    all_flows = all_flows.transpose(1,0,2,3)
    
    if view == 'xy':
        all_flows = all_flows.transpose(0,1,2,3)
        
    if view == 'xz':
        all_probs = all_probs.transpose(1,0,2)
        all_flows = all_flows.transpose(0,2,1,3) # the first channel is the flow!.
        
    if view == 'yz':
        all_probs = all_probs.transpose(1,2,0)
        all_flows = all_flows.transpose(0,2,3,1) # the first channel is the flow!.
        
        
    if (savefolder is not None) and (basename is not None):
        
        usegment3D_fio.write_pickle(os.path.join(savefolder, 
                                  basename+'_cellpose_flows_%s.pkl' %(str(view))), 
                                  {'flow': all_flows.astype(np.float32)})
        usegment3D_fio.write_pickle(os.path.join(savefolder, 
                                  basename+'_cellpose_probs_%s.pkl' %(str(view))), 
                                  {'prob': all_probs.astype(np.float32)})
        # write_pickle(os.path.join(savefolder, 
        #                           basename+'_cellpose_masks_%s.pkl' %(str(view))), 
        #                           {'mask': all_masks})
        spio.savemat(os.path.join(savefolder, 
                                  basename+'_cellpose_diam-styles_%s.mat' %(str(view))), 
                      {#'diam' : all_diams,
                      'style': all_styles,
                      'diam_range':auto_diam, 
                      'diam_score':auto_diam[1], 
                      'best_diam':auto_diam[2]}, do_compression=True)
        
    return auto_diam, all_probs, all_flows, all_styles


# make the equivalent below but for the indirect method which is more convoluted. 
def aggregate_2D_to_3D_segmentation_direct_method(probs, gradients, params,
                                                  precombined_binary = None, # this is to allow for using externally computed / combined versions. 
                                                  precombined_gradients = None,
                                                  savefolder=None,
                                                  basename=None):
    
    import numpy as np 
    import scipy.ndimage as ndimage
    import skimage.filters as skfilters
    import skimage.morphology as skmorph
    import os 
    
    assert(len(probs) == 3)
    assert(len(gradients) == 3)
    
    """
    First derive the binary, if this is not presupplied.
    """
    
    if precombined_binary is None:
        
        masks = list(probs) # split this. 
        
        # clip the 
        masks = [np.clip(mask, -88.72, 88.72) for mask in masks if len(mask)>0]
        
        prob_mask = params['combine_cell_probs']['cellpose_prob_mask']
        
        if prob_mask:
            masks = [1./(1.+np.exp(-mask)) for mask in masks if len(mask)>0]
            
        # combine the mask.
        mask_combine  = usegment3D_filters.var_combine(masks, 
                                                       ksize=params['combine_cell_probs']['ksize'],  # awesome... high ksize is good 
                                                       alpha = params['combine_cell_probs']['alpha'], # was 1.0 
                                                       eps=params['combine_cell_probs']['eps'])
        
        del masks # free up space.
        
        if params['combine_cell_probs']['smooth_sigma'] > 0:
            mask_combine = ndimage.gaussian_filter(mask_combine, sigma=params['combine_cell_probs']['smooth_sigma']) # do less smoothing. 3 is prob still too much. 
        
        if params['combine_cell_probs']['prob_thresh'] is None:
            prob_thresh = skfilters.threshold_multiotsu(mask_combine, params['combine_cell_probs']['threshold_n_levels'])[params['combine_cell_probs']['threshold_level']]
            
            if params['combine_cell_probs']['apply_one_d_p_thresh'] > 0:
                prob_thresh = int(prob_thresh*10)/10.
                
            # enforce a minimum since statistically we can be skewed ... 
            prob_thresh = np.maximum(params['combine_cell_probs']['min_prob_thresh'], prob_thresh)
            print('Automatric binary thresholding: ', prob_thresh)
            
            mask_binary = mask_combine >= prob_thresh
        else:
            mask_binary = mask_combine >= params['combine_cell_probs']['prob_thresh']
        
        
        # """
        # We can use a guided filter to try to refine!. 
        # """
        if params['postprocess_binary']['binary_closing'] > 0:
            mask_binary = skmorph.binary_closing(mask_binary, skmorph.ball(params['postprocess_binary']['binary_closing']))
        if params['postprocess_binary']['remove_small_objects'] > 0:
            mask_binary = skmorph.remove_small_objects(mask_binary, 
                                                       min_size=params['postprocess_binary']['remove_small_objects'],
                                                       connectivity=2)
        if params['postprocess_binary']['binary_dilation'] > 0:
            mask_binary = skmorph.binary_dilation(mask_binary, skmorph.ball(params['postprocess_binary']['binary_dilation']))
        if params['postprocess_binary']['binary_fill_holes'] > 0:
            mask_binary = ndimage.binary_fill_holes(mask_binary)
        if params['postprocess_binary']['extra_erode'] > 0:
            mask_binary = skmorph.binary_erosion(mask_binary, skmorph.ball(params['postprocess_binary']['extra_erode']))
    
    else:
            
        mask_binary = precombined_binary.copy()
        mask_combine = np.zeros(mask_binary.shape, dtype=np.float32); mask_combine[:] = np.nan
    
    
    """
    Second combining the gradients. var_combine, shouldn't matter about 0. 
    """
    im_shape = mask_binary.shape
    
    if precombined_gradients is None:
        # derive 
        mask_gradients = []
        
        for gradient_ii, gradient in enumerate(gradients):
            if len(gradient) > 0:
                gradient = np.concatenate([np.zeros(im_shape, dtype=np.float32)[None,...], 
                                                    gradient], axis=0)
                if gradient_ii == 0: # xy 
                    gradient = gradient[[0,1,2],...].copy()
                if gradient_ii == 1: # xz 
                    gradient = gradient[[1,0,2],...].copy()
                if gradient_ii == 2: # xz
                    gradient = gradient[[1,2,0],...].copy()
            else:
                gradient = [] # create a zero array. 
            mask_gradients.append(gradient)
            
        """
        Do filtering before combining by content. 
        """
        # do xy
        if len(mask_gradients[0])>0 and len(mask_gradients[1])>0:
            dx = usegment3D_filters.var_combine([ndimage.gaussian_filter(mask_gradients[0][2], sigma=params['combine_cell_gradients']['smooth_sigma']), 
                                                 ndimage.gaussian_filter(mask_gradients[1][2], sigma=params['combine_cell_gradients']['smooth_sigma'])],
                                                ksize=params['combine_cell_gradients']['ksize'],
                                                alpha=params['combine_cell_gradients']['alpha'])
        elif (len(mask_gradients[0])==0) and (len(mask_gradients[1])>0):
            dx = ndimage.gaussian_filter(mask_gradients[1][2], sigma=params['combine_cell_gradients']['smooth_sigma'])
        elif len(mask_gradients[0])>0 and len(mask_gradients[1])==0:
            dx = ndimage.gaussian_filter(mask_gradients[0][2], sigma=params['combine_cell_gradients']['smooth_sigma'])
        else:
            dx = np.zeros(im_shape, dtype=np.float32)
            
        
        if len(mask_gradients[0])>0 and len(mask_gradients[2])>0:
            dy = usegment3D_filters.var_combine([ndimage.gaussian_filter(mask_gradients[0][1], sigma=params['combine_cell_gradients']['smooth_sigma']), 
                                                  ndimage.gaussian_filter(mask_gradients[2][1], sigma=params['combine_cell_gradients']['smooth_sigma'])],
                                                ksize=params['combine_cell_gradients']['ksize'],
                                                alpha=params['combine_cell_gradients']['alpha'])
        elif len(mask_gradients[0])==0 and len(mask_gradients[2])>0:
            dy = ndimage.gaussian_filter(mask_gradients[2][1], sigma=params['combine_cell_gradients']['smooth_sigma'])
        elif len(mask_gradients[0])>0 and len(mask_gradients[2])==0:
            dy = ndimage.gaussian_filter(mask_gradients[0][1], sigma=params['combine_cell_gradients']['smooth_sigma'])
        else:
            dy = np.zeros(im_shape, dtype=np.float32)
            
        if len(mask_gradients[1])>0 and len(mask_gradients[2])>0:
            dz = usegment3D_filters.var_combine([ndimage.gaussian_filter(mask_gradients[1][0], sigma=1), 
                                                  ndimage.gaussian_filter(mask_gradients[2][0], sigma=1)],
                                                ksize=params['combine_cell_gradients']['ksize'],
                                                alpha=params['combine_cell_gradients']['alpha'])
        elif len(mask_gradients[1])==0 and len(mask_gradients[2])>0:
            dz = ndimage.gaussian_filter(mask_gradients[2][0], sigma=params['combine_cell_gradients']['smooth_sigma'])
        elif len(mask_gradients[1])>0 and len(mask_gradients[2])==0:
            dz = ndimage.gaussian_filter(mask_gradients[1][0], sigma=params['combine_cell_gradients']['smooth_sigma'])
        else:
            dz = np.zeros(im_shape, dtype=np.float32)
            
    
        del mask_gradients # free up space
        
        """
        To do: further smoothing 
        """
    
        dx = ndimage.gaussian_filter(dx, sigma=1.)
        dy = ndimage.gaussian_filter(dy, sigma=1.)
        dz = ndimage.gaussian_filter(dz, sigma=1.)
        labels_gradients = np.concatenate([ dz[None,...], 
                                            dy[None,...], 
                                            dx[None,...] ], axis=0)
        
        del dx, dy, dz 
    
        """
        Normalize the combined gradients. 
        """
        labels_gradients =labels_gradients / (np.linalg.norm(labels_gradients, axis=0)[None,...]+1e-20) 
        labels_gradients = labels_gradients.transpose(1,2,3,0).astype(np.float32) # put in the orientation expected by gradient descent. 
    
    else:
        
        labels_gradients = precombined_gradients.copy()
        
        if len(labels_gradients) == 3: 
            labels_gradients = labels_gradients / (np.linalg.norm(labels_gradients, axis=0)[None,...]+1e-20) 
            labels_gradients = labels_gradients.transpose(1,2,3,0).astype(np.float32) # put in the orientation expected by gradient descent. 
        else:
            labels_gradients = labels_gradients / (np.linalg.norm(labels_gradients, axis=-1)[...,None]+1e-20) 
    
    
    if (savefolder is not None) and (basename is not None):
        savepicklefile = os.path.join(savefolder, basename+'_combined2D_3D_probs_gradients_outputs.pkl')
        usegment3D_fio.write_pickle(savepicklefile, 
                                    {'binary':mask_binary, 
                                     'flow':labels_gradients,
                                     'prob':mask_combine})
        
    """
    Perform gradient descent aggregation with flag for MP. 
    """ 
    if params['gradient_descent']['do_mp']:
        labels_3D_watershed, cell_seg_connected, tracks, votes_grid_acc = usegment3D_watershed.mp_gradient_watershed3D_binary(mask_binary>0, 
                                                                                                                    gradient_img = labels_gradients,
                                                                                                                    tile_shape=params['gradient_descent']['tile_shape'],
                                                                                                                    tile_aspect=params['gradient_descent']['tile_aspect'],
                                                                                                                    tile_overlap_ratio = params['gradient_descent']['tile_overlap_ratio'],
                                                                                                                    momenta=params['gradient_descent']['momenta'], 
                                                                                                                    divergence_rescale=False, # i don't think this was defined correct helps.... ----> this actually seems to help by putting random fiducials...... (not actually stable..) 
                                                                                                                    smooth_sigma=params['connected_component']['smooth_sigma'], # use a larger for weird shapes!.  
                                                                                                                    smooth_gradient=1., # this has no effect except divergence_rescale=True
                                                                                                                    delta=params['gradient_descent']['delta'], 
                                                                                                                    n_iter=params['gradient_descent']['n_iter'],
                                                                                                                    gradient_decay=params['gradient_descent']['gradient_decay'],
                                                                                                                    interp = params['gradient_descent']['interp'],
                                                                                                                    binary_mask_gradient=params['gradient_descent']['binary_mask_gradient'],
                                                                                                                    eps=params['gradient_descent']['eps'], 
                                                                                                                    min_area=params['connected_component']['min_area'],
                                                                                                                    thresh_factor=params['connected_component']['thresh_factor'],
                                                                                                                    mask=None,
                                                                                                                    use_connectivity=params['gradient_descent']['use_connectivity'],
                                                                                                                    connectivity_alpha=params['gradient_descent']['connectivity_alpha'],
                                                                                                                    renorm_gradient=params['gradient_descent']['renorm_gradient'],
                                                                                                                    debug_viz=False) # can't visualize. 
    else:
        labels_3D_watershed, cell_seg_connected, tracks, votes_grid_acc = usegment3D_watershed.gradient_watershed3D_binary(mask_binary>0, 
                                                                                                                    gradient_img = labels_gradients,
                                                                                                                    momenta=params['gradient_descent']['momenta'], 
                                                                                                                    divergence_rescale=False, # i don't think this was defined correct helps.... ----> this actually seems to help by putting random fiducials...... (not actually stable..) 
                                                                                                                    smooth_sigma=params['connected_component']['smooth_sigma'], # use a larger for weird shapes!.  
                                                                                                                    smooth_gradient=1., # this has no effect except divergence_rescale=True
                                                                                                                    delta=params['gradient_descent']['delta'], 
                                                                                                                    n_iter=params['gradient_descent']['n_iter'],
                                                                                                                    gradient_decay=params['gradient_descent']['gradient_decay'],
                                                                                                                    interp = params['gradient_descent']['interp'],
                                                                                                                    binary_mask_gradient=params['gradient_descent']['binary_mask_gradient'],
                                                                                                                    eps=params['gradient_descent']['eps'], 
                                                                                                                    min_area=params['connected_component']['min_area'],
                                                                                                                    thresh_factor=params['connected_component']['thresh_factor'],
                                                                                                                    mask=None,
                                                                                                                    use_connectivity=params['gradient_descent']['use_connectivity'],
                                                                                                                    connectivity_alpha=params['gradient_descent']['connectivity_alpha'],
                                                                                                                    renorm_gradient=params['gradient_descent']['renorm_gradient'],
                                                                                                                    debug_viz=params['gradient_descent']['debug_viz'],
                                                                                                                    sampling=params['gradient_descent']['sampling'],
                                                                                                                    track_percent=params['gradient_descent']['track_percent'], 
                                                                                                                    rand_seed=params['gradient_descent']['rand_seed'], 
                                                                                                                    ref_initial_color_img=params['gradient_descent']['ref_initial_color_img'],
                                                                                                                    ref_alpha=params['gradient_descent']['ref_alpha'], 
                                                                                                                    saveplotsfolder= params['gradient_descent']['saveplotsfolder'],
                                                                                                                    viewinit=params['gradient_descent']['viewinit']) # 50 seems not enough for 3D... as it is very connected. 
        
    if (savefolder is not None) and (basename is not None):
        # save the final segmentation and in color. 
        usegment3D_fio.save_segmentation(os.path.join(savefolder,
                                                      basename+'_2D-to-3D_aggregated_u-Segment3D_direct_method.tif'), 
                                         labels_3D_watershed)
    
    return labels_3D_watershed, (mask_combine, labels_gradients)



# make the equivalent below but for the indirect method which is more convoluted. 
def aggregate_2D_to_3D_segmentation_indirect_method(segmentations, 
                                                    params,
                                                    img_xy_shape,
                                                    precomputed_binary = None, # this is to allow for using externally computed / combined versions. 
                                                    savefolder=None,
                                                    basename=None):
    
    import numpy as np 
    import scipy.ndimage as ndimage
    import skimage.filters as skfilters
    import skimage.morphology as skmorph
    import os 

    # check we always pass the correct number of masks    
    assert(len(segmentations) == 3)
    
    """
    First derive the binary, if this is not presupplied.
    """
    
    masks = [] # split this. 
    
    for mask_ii, mask in enumerate(segmentations):
        
        if len(mask) > 0:
            masks.append(mask) # get the foreground
        else:
            if mask_ii == 0:
                mask = np.zeros(img_xy_shape, dtype=np.uint16)
            if mask_ii == 1:
                mask = np.zeros(np.hstack(img_xy_shape)[[1,0,2]], dtype=np.uint16)
            if mask_ii == 2:
                mask = np.zeros(np.hstack(img_xy_shape)[[2,0,1]], dtype=np.uint16)
            masks.append(mask)
            
    # combine the mask.
    sigma = params['indirect_method']['smooth_binary']
    
    if precomputed_binary is None:
        if sigma > 0:
            mask_combine = usegment3D_filters.var_combine([ndimage.gaussian_filter((masks[0]>0)*1., sigma=sigma), 
                                                           ndimage.gaussian_filter((masks[1]>0).transpose(1,0,2)*1., sigma=sigma), 
                                                           ndimage.gaussian_filter((masks[2]>0).transpose(1,2,0)*1., sigma=sigma)], 
                                                           ksize=params['combine_cell_probs']['ksize'],  # awesome... high ksize is good 
                                                           alpha = params['combine_cell_probs']['alpha'], # was 1.0 
                                                           eps=params['combine_cell_probs']['eps'])
        else:
            mask_combine = usegment3D_filters.var_combine([(masks[0]>0)*1., 
                                                           (masks[1]>0).transpose(1,0,2)*1., 
                                                           (masks[2]>0).transpose(1,2,0)*1.], 
                                                           ksize=params['combine_cell_probs']['ksize'],  # awesome... high ksize is good 
                                                           alpha = params['combine_cell_probs']['alpha'], # was 1.0 
                                                           eps=params['combine_cell_probs']['eps'])
        
        #  rescale to combat only have partial views? 
        mask_combine = np.clip(mask_combine / (np.max(mask_combine)+1e-12), 0, 1) # rescale to 0-1
        
        # further smooth the combined masks. 
        if params['combine_cell_probs']['smooth_sigma'] > 0:
            mask_combine = ndimage.gaussian_filter(mask_combine, sigma=params['combine_cell_probs']['smooth_sigma']) # do less smoothing. 3 is prob still too much. 
        
        if params['combine_cell_probs']['prob_thresh'] is None:
            prob_thresh = skfilters.threshold_multiotsu(mask_combine, params['combine_cell_probs']['threshold_n_levels'])[params['combine_cell_probs']['threshold_level']]
            
            if params['combine_cell_probs']['apply_one_d_p_thresh'] > 0:
                prob_thresh = int(prob_thresh*10)/10.
                
            # enforce a minimum since statistically we can be skewed ... 
            prob_thresh = np.maximum(params['combine_cell_probs']['min_prob_thresh'], prob_thresh)
            print('Automatric binary thresholding: ', prob_thresh)
            
            mask_binary = mask_combine >= prob_thresh
        else:
            mask_binary = mask_combine >= params['combine_cell_probs']['prob_thresh']
        
        # """
        # We can use a guided filter to try to refine!. 
        # """
        if params['postprocess_binary']['binary_closing'] > 0:
            mask_binary = skmorph.binary_closing(mask_binary, skmorph.ball(params['postprocess_binary']['binary_closing']))
        if params['postprocess_binary']['remove_small_objects'] > 0:
            mask_binary = skmorph.remove_small_objects(mask_binary, 
                                                       min_size=params['postprocess_binary']['remove_small_objects'],
                                                       connectivity=2)
        if params['postprocess_binary']['binary_dilation'] > 0:
            mask_binary = skmorph.binary_dilation(mask_binary, skmorph.ball(params['postprocess_binary']['binary_dilation']))
        if params['postprocess_binary']['binary_fill_holes'] > 0:
            mask_binary = ndimage.binary_fill_holes(mask_binary)
        if params['postprocess_binary']['extra_erode'] > 0:
            mask_binary = skmorph.binary_erosion(mask_binary, skmorph.ball(params['postprocess_binary']['extra_erode']))

    else:
            
        mask_binary = precomputed_binary.copy()
        mask_combine = np.zeros(mask_binary.shape, dtype=np.float32); mask_combine[:] = np.nan
    
    """
    Second compute the gradients using the selected distance transform and combine into 3D gradients. var_combine, shouldn't matter about 0. 
    """

    mask_gradients = []
        
    for mask_ii, mask in enumerate(masks):
        
        if mask.max() > 0: 
            
            if params['indirect_method']['dtform_method'] == 'edt':
                # if euclidean then this is simple
                gradient = usegment3D_flows.distance_transform_labels_fast(mask*1) # ensure int. 
                gradient = np.array([np.array(np.gradient(dist_slice)) for dist_slice in gradient]).transpose(1,0,2,3) # this makes the vectors first dimension
                gradient = gradient / (np.linalg.norm(gradient, axis=0)[None,...] + 1e-20)
            
            else:
                skel_guide_image = usegment3D_flows.distance_transform_labels_fast(mask*1)
                
                n_processes = params['indirect_method']['n_cpu']
                if n_processes is None:
                    import multiprocess as mp 
                    n_processes = (mp.cpu_count() - 1) // 2 # halve this? 
                
                gradient = usegment3D_flows.distance_centroid_tform_flow_labels2D_dask(mask*1, 
                                                                                        dtform_method=params['indirect_method']['dtform_method'],
                                                                                        guide_image=skel_guide_image,
                                                                                        # guide_image = None, # this is not currently used. 
                                                                                        fixed_point_percentile=params['indirect_method']['edt_fixed_point_percentile'], 
                                                                                        n_processes=n_processes,
                                                                                        iter_factor =  params['indirect_method']['iter_factor'], 
                                                                                        smooth_skel_sigma=params['indirect_method']['smooth_skel_sigma'],
                                                                                        power_dist=params['indirect_method']['power_dist'])
                gradient = gradient[:,1:].transpose(1,0,2,3).copy() # this should now be the same size as the edt. 
                
            if mask_ii == 0: #(x,y)
                gradient = gradient.transpose(0,1,2,3).astype(np.float32)
            if mask_ii == 1: #(x,z)
                gradient = gradient.transpose(0,2,1,3).astype(np.float32)
            if mask_ii == 2: #(y,z)
                gradient = gradient.transpose(0,2,3,1).astype(np.float32)
                
            gradient = np.concatenate([np.zeros(img_xy_shape, dtype=np.float32)[None,...], 
                                                gradient], axis=0)
            
            # perform the relevant switches. 
            if mask_ii == 0: # xy 
                gradient = gradient[[0,1,2],...].copy()
            if mask_ii == 1: # xz 
                gradient = gradient[[1,0,2],...].copy()
            if mask_ii == 2: # xz
                gradient = gradient[[1,2,0],...].copy()
                
            mask_gradients.append(gradient) # make sure to append 
        else:
            mask_gradients.append([]) # everything is zero so nothing to transpose and flip. 
            
    
    """
    Do filtering before combining by content. 
    """
    # do xy
    if len(mask_gradients[0])>0 and len(mask_gradients[1])>0:
        dx = usegment3D_filters.var_combine([ndimage.gaussian_filter(mask_gradients[0][2], sigma=params['combine_cell_gradients']['smooth_sigma']), 
                                             ndimage.gaussian_filter(mask_gradients[1][2], sigma=params['combine_cell_gradients']['smooth_sigma'])],
                                            ksize=params['combine_cell_gradients']['ksize'],
                                            alpha=params['combine_cell_gradients']['alpha'])
    elif (len(mask_gradients[0])==0) and (len(mask_gradients[1])>0):
        dx = ndimage.gaussian_filter(mask_gradients[1][2], sigma=params['combine_cell_gradients']['smooth_sigma'])
    elif len(mask_gradients[0])>0 and len(mask_gradients[1])==0:
        dx = ndimage.gaussian_filter(mask_gradients[0][2], sigma=params['combine_cell_gradients']['smooth_sigma'])
    else:
        dx = np.zeros(img_xy_shape, dtype=np.float32)
        
    
    if len(mask_gradients[0])>0 and len(mask_gradients[2])>0:
        dy = usegment3D_filters.var_combine([ndimage.gaussian_filter(mask_gradients[0][1], sigma=params['combine_cell_gradients']['smooth_sigma']), 
                                              ndimage.gaussian_filter(mask_gradients[2][1], sigma=params['combine_cell_gradients']['smooth_sigma'])],
                                            ksize=params['combine_cell_gradients']['ksize'],
                                            alpha=params['combine_cell_gradients']['alpha'])
    elif len(mask_gradients[0])==0 and len(mask_gradients[2])>0:
        dy = ndimage.gaussian_filter(mask_gradients[2][1], sigma=params['combine_cell_gradients']['smooth_sigma'])
    elif len(mask_gradients[0])>0 and len(mask_gradients[2])==0:
        dy = ndimage.gaussian_filter(mask_gradients[0][1], sigma=params['combine_cell_gradients']['smooth_sigma'])
    else:
        dy = np.zeros(img_xy_shape, dtype=np.float32)
        
    if len(mask_gradients[1])>0 and len(mask_gradients[2])>0:
        dz = usegment3D_filters.var_combine([ndimage.gaussian_filter(mask_gradients[1][0], sigma=1), 
                                              ndimage.gaussian_filter(mask_gradients[2][0], sigma=1)],
                                            ksize=params['combine_cell_gradients']['ksize'],
                                            alpha=params['combine_cell_gradients']['alpha'])
    elif len(mask_gradients[1])==0 and len(mask_gradients[2])>0:
        dz = ndimage.gaussian_filter(mask_gradients[2][0], sigma=params['combine_cell_gradients']['smooth_sigma'])
    elif len(mask_gradients[1])>0 and len(mask_gradients[2])==0:
        dz = ndimage.gaussian_filter(mask_gradients[1][0], sigma=params['combine_cell_gradients']['smooth_sigma'])
    else:
        dz = np.zeros(img_xy_shape, dtype=np.float32)

    del mask_gradients # free up space
        
    """
    To do: ** further smoothing **
    """
    dx = ndimage.gaussian_filter(dx, sigma=1.)
    dy = ndimage.gaussian_filter(dy, sigma=1.)
    dz = ndimage.gaussian_filter(dz, sigma=1.)
    labels_gradients = np.concatenate([ dz[None,...], 
                                        dy[None,...], 
                                        dx[None,...] ], axis=0)
    
    del dx, dy, dz 
    
    """
    Normalize the combined gradients. 
    """
    labels_gradients =labels_gradients / (np.linalg.norm(labels_gradients, axis=0)[None,...]+1e-20) 
    labels_gradients = labels_gradients.transpose(1,2,3,0).astype(np.float32) # put in the orientation expected by gradient descent. 
    

    if (savefolder is not None) and (basename is not None):
        savepicklefile = os.path.join(savefolder, basename+'_combined2D_3D_probs_gradients_outputs.pkl')
        usegment3D_fio.write_pickle(savepicklefile, 
                                    {'binary':mask_binary, 
                                     'flow':labels_gradients,
                                     'prob':mask_combine})
        
    """
    Perform gradient descent aggregation with flag for MP. (and this is identical to that of the direct method. )
    """ 
    if params['gradient_descent']['do_mp']:
        labels_3D_watershed, cell_seg_connected, tracks, votes_grid_acc = usegment3D_watershed.mp_gradient_watershed3D_binary(mask_binary>0, 
                                                                                                                    gradient_img = labels_gradients,
                                                                                                                    tile_shape=params['gradient_descent']['tile_shape'],
                                                                                                                    tile_aspect=params['gradient_descent']['tile_aspect'],
                                                                                                                    tile_overlap_ratio = params['gradient_descent']['tile_overlap_ratio'],
                                                                                                                    momenta=params['gradient_descent']['momenta'], 
                                                                                                                    divergence_rescale=False, # i don't think this was defined correct helps.... ----> this actually seems to help by putting random fiducials...... (not actually stable..) 
                                                                                                                    smooth_sigma=params['connected_component']['smooth_sigma'], # use a larger for weird shapes!.  
                                                                                                                    smooth_gradient=1., # this has no effect except divergence_rescale=True
                                                                                                                    delta=params['gradient_descent']['delta'], 
                                                                                                                    n_iter=params['gradient_descent']['n_iter'],
                                                                                                                    gradient_decay=params['gradient_descent']['gradient_decay'],
                                                                                                                    interp = params['gradient_descent']['interp'],
                                                                                                                    binary_mask_gradient=params['gradient_descent']['binary_mask_gradient'],
                                                                                                                    eps=params['gradient_descent']['eps'], 
                                                                                                                    min_area=params['connected_component']['min_area'],
                                                                                                                    thresh_factor=params['connected_component']['thresh_factor'],
                                                                                                                    mask=None,
                                                                                                                    use_connectivity=params['gradient_descent']['use_connectivity'],
                                                                                                                    connectivity_alpha=params['gradient_descent']['connectivity_alpha'],
                                                                                                                    renorm_gradient=params['gradient_descent']['renorm_gradient'],
                                                                                                                    debug_viz=False) # can't visualize. 
    else:
        labels_3D_watershed, cell_seg_connected, tracks, votes_grid_acc = usegment3D_watershed.gradient_watershed3D_binary(mask_binary>0, 
                                                                                                                    gradient_img = labels_gradients,
                                                                                                                    momenta=params['gradient_descent']['momenta'], 
                                                                                                                    divergence_rescale=False, # i don't think this was defined correct helps.... ----> this actually seems to help by putting random fiducials...... (not actually stable..) 
                                                                                                                    smooth_sigma=params['connected_component']['smooth_sigma'], # use a larger for weird shapes!.  
                                                                                                                    smooth_gradient=1., # this has no effect except divergence_rescale=True
                                                                                                                    delta=params['gradient_descent']['delta'], 
                                                                                                                    n_iter=params['gradient_descent']['n_iter'],
                                                                                                                    gradient_decay=params['gradient_descent']['gradient_decay'],
                                                                                                                    interp = params['gradient_descent']['interp'],
                                                                                                                    binary_mask_gradient=params['gradient_descent']['binary_mask_gradient'],
                                                                                                                    eps=params['gradient_descent']['eps'], 
                                                                                                                    min_area=params['connected_component']['min_area'],
                                                                                                                    thresh_factor=params['connected_component']['thresh_factor'],
                                                                                                                    mask=None,
                                                                                                                    use_connectivity=params['gradient_descent']['use_connectivity'],
                                                                                                                    connectivity_alpha=params['gradient_descent']['connectivity_alpha'],
                                                                                                                    renorm_gradient=params['gradient_descent']['renorm_gradient'],
                                                                                                                    debug_viz=params['gradient_descent']['debug_viz'],
                                                                                                                    sampling=params['gradient_descent']['sampling'],
                                                                                                                    track_percent=params['gradient_descent']['track_percent'], 
                                                                                                                    rand_seed=params['gradient_descent']['rand_seed'], 
                                                                                                                    ref_initial_color_img=params['gradient_descent']['ref_initial_color_img'],
                                                                                                                    ref_alpha=params['gradient_descent']['ref_alpha'], 
                                                                                                                    saveplotsfolder= params['gradient_descent']['saveplotsfolder'],
                                                                                                                    viewinit=params['gradient_descent']['viewinit']) # 50 seems not enough for 3D... as it is very connected. 
        
    if (savefolder is not None) and (basename is not None):
        # save the final segmentation and in color. 
        usegment3D_fio.save_segmentation(os.path.join(savefolder,
                                                      basename+'_2D-to-3D_aggregated_u-Segment3D_direct_method.tif'), 
                                         labels_3D_watershed)
    
    return labels_3D_watershed, (mask_combine, labels_gradients)


def postprocess_3D_cell_segmentation(segmentation,
                                     aggregation_params,
                                     postprocess_params,
                                     cell_gradients=None,
                                     savefolder=None,
                                     basename=None):
    
    
    from cellpose.utils import fill_holes_and_remove_small_masks
    import scipy.ndimage as ndimage
    import skimage.measure as skmeasure 
    
    
    segmentation_filt = segmentation.copy()
    
    flow_consistency_outputs = []
    
    """
    1. size filter. 
    """
    params = postprocess_params.copy()
    
    # minimum size filter. 
    if params['size_filters']['min_size']>0:
        # remove small regions and keep one label? 
        segmentation_filt = fill_holes_and_remove_small_masks(segmentation, 
                                                              min_size=params['size_filters']['min_size']) ### this should be implemented to not depend on cellpose!. 
    # maximum size filter. 
    if params['size_filters']['max_size'] is not None:

        uniq_labels = np.setdiff1d(np.unique(segmentation_filt), 0) 
        regions = skmeasure.regionprops(segmentation_filt)
        
        areas = np.hstack([re.area for re in regions])
        area_thresh = params['size_filters']['max_size']
        
        if np.sum(areas>area_thresh): # there are regions greater than this threshold 
            remove_labels_indices = np.arange(len(uniq_labels))[areas>area_thresh]
            # remove_labels = uniq_labels[remove_labels_indices]
            
            for rr in remove_labels_indices:
                coords = regions[rr].coords
                segmentation_filt[coords[:,0], coords[:,1], coords[:,2]] = 0 
                
    
    segmentation_filt = usegment3D_filters.largest_component_vol_labels_fast(segmentation_filt, connectivity=1)
    
    
    """
    2. flow consistency filter. 
    """
    # In general we do this except 
    if params['flow_consistency']['do_flow_remove'] and cell_gradients is not None: 
        
        # do the multiprocess. 
        import multiprocess as mp # this version allows pickling. 
        # import multiprocessing as mp
        
        dtform_method = params['flow_consistency']['dtform_method']
        n_cpu = params['flow_consistency']['n_cpu']
        
        if n_cpu is None:
            n_cpu = np.maximum(mp.cpu_count() - 1,1)
            
        
        tmp = segmentation_filt.copy()

        if dtform_method != 'edt':
            ### we should run the below if dtform_method is not edt... otherwise edt is much much simpler. 
            guide = usegment3D_flows.distance_transform_labels_fast(tmp, n_threads=n_cpu)
            
            def _compute_cellflow_view(slice_id): # can't really pickle this. 
                # out = grad_flows.distance_centroid_tform_flow_labels2D()
                
                if dtform_method == 'edt':
                    # this is just the guide
                    out = usegment3D_flows.distance_transform_labels_fast(tmp[slice_id], n_threads=n_cpu)
                    out_flow = np.array(np.gradient(out))
                    out_flow = out_flow / (np.linalg.norm(out, axis=0)[None,...] + 1e-20)
                    out = np.concatenate([out[None,...], out_flow], axis=0)
                else:    
                    out = usegment3D_flows.distance_centroid_tform_flow_labels2D(tmp[slice_id], 
                                                                            dtform_method=dtform_method,
                                                                            guide_img=guide[slice_id], 
                                                                            fixed_point_percentile=params['flow_consistency']['edt_fixed_point_percentile'], 
                                                                            iter_factor=params['flow_consistency']['iter_factor'],
                                                                            smooth_skel_sigma=params['flow_consistency']['smooth_skel_sigma'],
                                                                            power_dist=params['flow_consistency']['power_dist'])
                return out 
            
            
            print('start xy')
            # do xy first. 
            with mp.Pool(n_cpu) as pool:
            # with ThreadPool(mp.cpu_count()-1) as pool: 
                # res = pool.map(compute_cellflow_view_xz, range(0, len(labels_3D_watershed_filt.transpose(1,0,2))))
                gradient_xy = pool.map(_compute_cellflow_view, range(0, len(tmp)))
                gradient_xy = np.array(gradient_xy, dtype=np.float32) #.transpose(1,0,2,3)
                gradient_xy = gradient_xy[:,1:]
                gradient_xy = np.concatenate([np.zeros_like(gradient_xy[:,1])[:,None,...], gradient_xy], axis=1)
                gradient_xy = gradient_xy[:,[0,1,2],...] #.copy() # we must flip the channels!. 
            print('finish xy')
                
            # do xz 
            tmp = segmentation_filt.transpose(1,0,2).copy()
            guide = usegment3D_flows.distance_transform_labels_fast(tmp, n_threads=n_cpu)
            
            print('start xz')
            with mp.Pool(n_cpu) as pool:
            # with ThreadPool(mp.cpu_count()-1) as pool: 
                # res = pool.map(compute_cellflow_view_xz, range(0, len(labels_3D_watershed_filt.transpose(1,0,2))))
                gradient_xz = pool.map(_compute_cellflow_view, range(0, len(tmp)))
                gradient_xz = np.array(gradient_xz,dtype=np.float32)#.transpose(1,0,2,3)
                gradient_xz = gradient_xz[:,1:].copy()
                gradient_xz = gradient_xz.transpose( 2,1,0,3 )
                gradient_xz = np.concatenate([np.zeros_like(gradient_xz[:,1])[:,None,...], gradient_xz], axis=1)
                gradient_xz = gradient_xz[:,[1,0,2],...]#.copy() # we must flip the channels!. 
            print('finish xz')
                
            # do yz 
            tmp = segmentation_filt.transpose(2,0,1).copy()
            guide = usegment3D_flows.distance_transform_labels_fast(tmp, n_threads=n_cpu)
            
            print('start yz')
            with mp.Pool(n_cpu) as pool:
            # with ThreadPool(mp.cpu_count()-1) as pool:
                # res = pool.map(compute_cellflow_view_xz, range(0, len(labels_3D_watershed_filt.transpose(1,0,2))))
                gradient_yz = pool.map(_compute_cellflow_view, range(0, len(tmp)))
                gradient_yz = np.array(gradient_yz,dtype=np.float32)#.transpose(1,0,2,3)
                gradient_yz = gradient_yz[:,1:] ; gradient_yz = gradient_yz.transpose(2,1,3,0) 
                gradient_yz = np.concatenate([np.zeros_like(gradient_yz[:,1])[:,None,...], gradient_yz], axis=1)
                gradient_yz = gradient_yz[:,[1,2,0],...] #.copy() # we must flip the channels!. 
            print('finish yz')
        
        else:
            # do xy
            tmp = segmentation_filt.copy()
            gradient_xy = usegment3D_flows.distance_transform_labels_fast(tmp*1,n_threads=n_cpu)
            gradient_xy = np.array([np.array(np.gradient(dist_slice)) for dist_slice in gradient_xy])
            gradient_xy = gradient_xy / (np.linalg.norm(gradient_xy, axis=1)[:,None] + 1e-20)
            gradient_xy = np.concatenate([np.zeros_like(gradient_xy[:,1])[:,None,...], gradient_xy], axis=1)
            gradient_xy = gradient_xy[:,[0,1,2],...] #.copy() # we must flip the channels!. 
            print('finish xy')
            
            # do xz 
            tmp = segmentation_filt.transpose(1,0,2).copy()
            gradient_xz = usegment3D_flows.distance_transform_labels_fast(tmp*1, n_threads=n_cpu)
            gradient_xz = np.array([np.array(np.gradient(dist_slice)) for dist_slice in gradient_xz])
            gradient_xz = gradient_xz / (np.linalg.norm(gradient_xz, axis=1)[:,None] + 1e-20)
            gradient_xz = gradient_xz.transpose( 2,1,0,3 )
            gradient_xz = np.concatenate([np.zeros_like(gradient_xz[:,1])[:,None,...], gradient_xz], axis=1)
            gradient_xz = gradient_xz[:,[1,0,2],...]#.copy() # we must flip the channels!. 
            print('finish xz')
            
            # do yz 
            tmp = segmentation_filt.transpose(2,0,1).copy()
            gradient_yz = usegment3D_flows.distance_transform_labels_fast(tmp*1, n_threads=n_cpu)
            gradient_yz = np.array([np.array(np.gradient(dist_slice)) for dist_slice in gradient_yz])
            gradient_yz = gradient_yz / (np.linalg.norm(gradient_yz, axis=1)[:,None] + 1e-20)
            gradient_yz = gradient_yz.transpose(2,1,3,0) 
            gradient_yz = np.concatenate([np.zeros_like(gradient_yz[:,1])[:,None,...], gradient_yz], axis=1)
            gradient_yz = gradient_yz[:,[1,2,0],...] #.copy() # we must flip the channels!. 
            print('finish yz')
        
        """
        implement the combination. 
        """
        del tmp, guide
            
        dx = usegment3D_filters.var_combine([ndimage.gaussian_filter(gradient_xy[:,2], sigma=aggregation_params['combine_cell_gradients']['smooth_sigma']), 
                                       ndimage.gaussian_filter(gradient_xz[:,2], sigma=aggregation_params['combine_cell_gradients']['smooth_sigma'])],
                                      ksize=aggregation_params['combine_cell_gradients']['ksize'],
                                      alpha=aggregation_params['combine_cell_gradients']['alpha'])
        dy = usegment3D_filters.var_combine([ndimage.gaussian_filter(gradient_xy[:,1],sigma=aggregation_params['combine_cell_gradients']['smooth_sigma']), 
                                        ndimage.gaussian_filter(gradient_yz[:,1],sigma=aggregation_params['combine_cell_gradients']['smooth_sigma'])],
                                      ksize=aggregation_params['combine_cell_gradients']['ksize'], # we probably need to tune this up! for this imaging 
                                      alpha=aggregation_params['combine_cell_gradients']['alpha'])
        dz = usegment3D_filters.var_combine([ndimage.gaussian_filter(gradient_xz[:,0], sigma=aggregation_params['combine_cell_gradients']['smooth_sigma']), 
                                        ndimage.gaussian_filter(gradient_yz[:,0], sigma=aggregation_params['combine_cell_gradients']['smooth_sigma'])],
                                      ksize=aggregation_params['combine_cell_gradients']['ksize'],
                                      alpha=aggregation_params['combine_cell_gradients']['alpha'])
        
        # further implement the post filtering. 
        dx = ndimage.gaussian_filter(dx, sigma=1.)
        dy = ndimage.gaussian_filter(dy, sigma=1.)
        dz = ndimage.gaussian_filter(dz, sigma=1.)
        
        dzyx = np.concatenate([ dz[:,None,...], 
                                dy[:,None,...], 
                                dx[:,None,...] ], axis=1)
        del dx, dy, dz
        
        # do a little filtering. 
        dzyx = dzyx.transpose(1,0,2,3)
        dzyx = dzyx / (np.linalg.norm(dzyx, axis=0)[None,...] + 1e-20) # do a renormalize!. 


        print('finished computing gradients and removing bad flow masks ...')
        flow_errors=np.zeros(segmentation_filt.max())
        for i in range(dzyx.shape[0]): # over the 3 dimensions. # this should be fast. 
            flow_errors += ndimage.mean((dzyx[i] - cell_gradients.transpose(3,0,1,2)[i])**2, 
                                        segmentation_filt,
                                        index=np.arange(1, segmentation_filt.max()+1))
        flow_errors[np.isnan(flow_errors)] = 0 # zero out any nan entries!. 
            
        
        error_labels = flow_errors > params['flow_consistency']['flow_threshold']
        
        if np.sum(error_labels) > 0:
            # perform correction.     
            valid_labels = np.setdiff1d(np.unique(segmentation_filt), 0)
            reg_error_labels = np.arange(1, segmentation_filt.max()+1)[error_labels] # these are the labels to remove!. 
            reg_error_labels = np.intersect1d(reg_error_labels, valid_labels)
            
            reg_error_labels_indices = np.hstack([np.arange(len(valid_labels))[valid_labels==ll] for ll in reg_error_labels])
            
            # set these regions to 0. 
            mask_new = segmentation_filt.copy()
            mask_regionprops = skmeasure.regionprops(segmentation_filt) # use this. 
            
            for lab in reg_error_labels_indices: # so this need to be faster. but how ? 
                # mask_new[mask==lab] = 0
                coords = mask_regionprops[lab].coords
                mask_new[coords[:,0],
                         coords[:,1],
                         coords[:,2]] = 0
                
            segmentation_filt = mask_new.copy()
            flow_consistency_outputs.append([flow_errors, dzyx.transpose(1,2,3,0)]) # the transpose is to return in the same shape as cell gradients. 
            
            del mask_new
            
            # # check also the below: which might be easier to call. # wwe should use this method. 
            # segmentation_filt, flow_errors, ref_cell_gradients = grad_flows.remove_bad_flow_masks_3D(segmentation_filt, 
            #                                                                              cell_gradients.transpose(3,0,1,2),
            #                                                                              flow_threshold=flow_threshold, # get rid of only the very worst ones! 
            #                                                                              n_processes=mp.cpu_count-1) # set a generous threshold.
            
            # remove_bad_flow_masks_3D(mask, 
            #                  flow, 
            #                  flow_threshold=0.6,
            #                  dtform_method='cellpose_improve',  
            #                  fixed_point_percentile=0.01, 
            #                  n_processes=4,
            #                  power_dist=None,
            #                  alpha=0.5, 
            #                  smooth_sigma=0,
            #                  smooth_ds=1,
            #                  filter_scale = 1)
    """
    3. statistical size filter
    """
    
    if params['size_filters']['do_stats_filter']:
    
        uniq_labels = np.setdiff1d(np.unique(segmentation_filt), 0) 
        regions = skmeasure.regionprops(segmentation_filt)
        
        areas = np.hstack([re.area for re in regions])
        area_thresh = np.nanmean(areas) + params['size_filters']['max_size_factor']*np.nanstd(areas)
        
        if np.sum(areas>area_thresh):
            remove_labels_indices = np.arange(len(uniq_labels))[areas>area_thresh]
            # remove_labels = uniq_labels[remove_labels_indices]
            
            for rr in remove_labels_indices:
                coords = regions[rr].coords
                segmentation_filt[coords[:,0], coords[:,1], coords[:,2]] = 0 
                
        
    return segmentation_filt, flow_consistency_outputs
        
        
        
# # label diffusion and guided filtering 
def label_diffuse_3D_cell_segmentation_MP(segmentation,
                                          guide_image,
                                          params): # does this work as module or else, we may need to implement in main.... 
    
    import skimage.measure as skmeasure
    import multiprocess as mp 
    import numpy as np
    
    n_cpu = params['diffusion']['n_cpu']
    if n_cpu is None:
        n_cpu = np.maximum(mp.cpu_count() - 1,1)
    
    # guide_img must be in range 0, 1, enforce here. 
    guide_img = usegment3D_filters.normalize(guide_image, 
                                               pmin=params['guide_img']['pmin'], 
                                               pmax=params['guide_img']['pmax'], clip=True)
    
    """
    define the parameters upfront
    """
    refine_clamp = params['diffusion']['refine_clamp']
    refine_iters = params['diffusion']['refine_iters']
    refine_alpha = params['diffusion']['refine_alpha']
    noprogress_bool = params['diffusion']['noprogress_bool']
    affinity_type = params['diffusion']['affinity_type']
    
    multilabel_refine = params['diffusion']['multilabel_refine']
    
    
    # some setup. 
    pad_size = params['diffusion']['pad_size']
    
    pad = np.hstack([pad_size,pad_size,pad_size])
    padding = ((pad_size,pad_size), (pad_size,pad_size), (pad_size,pad_size))
    guide_img_pad = np.pad(guide_img, padding, mode='reflect')
    
    segmentation_pad = np.pad(segmentation, padding, mode='constant')*1 # maybe this will be fine!
    
    # print(guide_img_pad.shape, segmentation_pad.shape)
    # print(segmentation_pad.max())
    
    # find out the individual cell ids
    cells = skmeasure.regionprops(segmentation_pad)
    cells_id = np.setdiff1d(np.unique(segmentation_pad), 0 ) # exclude the background label
    
    # print(len(cells))
    bboxes = [cells[ii].bbox for ii in range(0, len(cells))]
    
    """
    define the internal function
    """
    def _refine_single_obj(obj_id):
        
        # bbox = cells[obj_id].bbox
        # area = cells[obj_id].area
        bbox = bboxes[obj_id]
        
        label_bbox = segmentation_pad[bbox[0]-pad[0]:bbox[3]+pad[0],
                                      bbox[1]-pad[1]:bbox[4]+pad[1],
                                      bbox[2]-pad[2]:bbox[5]+pad[2]]
        label_img = guide_img_pad[bbox[0]-pad[0]:bbox[3]+pad[0],
                                  bbox[1]-pad[1]:bbox[4]+pad[1],
                                  bbox[2]-pad[2]:bbox[5]+pad[2]]
        
        # if multilabel_refine:
        #     label_bbox_refine = usegment3D_filters.diffuse_labels3D(label_bbox*1, 
        #                                                                 label_img, clamp=refine_clamp,
        #                                                                 n_iter=refine_iters, # was 25
        #                                                                 alpha=refine_alpha, # was 0.25
        #                                                                 noprogress=noprogress_bool,
        #                                                                 affinity_type=affinity_type)
        #     label_coords = np.argwhere(label_bbox_refine==cells_id[obj_id]) # has to be the same id. 
        # else:
        label_bbox_refine = usegment3D_filters.diffuse_labels3D((label_bbox==cells_id[obj_id])*1, 
                                                                    label_img, clamp=refine_clamp,
                                                                    n_iter=refine_iters, # was 25
                                                                    alpha=refine_alpha, # was 0.25
                                                                    noprogress=noprogress_bool,
                                                                    affinity_type=affinity_type)
        label_coords = np.argwhere(label_bbox_refine>0) # where ever is positive binary 
        
        if len(label_coords)>0:
            label_coords = label_coords + (bbox[:3]-pad[:3])[None,:] # remove the artificial padding. 
        else:
            label_coords = [] # empty 
      
        # suppress printing. 
        # print(obj_id)
        return label_coords
    
    """
    execute the multiprocess
    """
    
    with mp.Pool(n_cpu) as pool: # potentially thread is faster? 
    #     with mp.Pool(n_procs-1) as pool:
    # # with ThreadPool(n_procs) as pool: #586 with multiprocess but dask then does work. 
        res = pool.map(_refine_single_obj, range(0, len(cells)))
        # res = np.concatenate(res, 1) # collapse all. 
        # res = res - offset[None,None,:] # subtract the global padding.... everything else shud be fine!. 
    # remove the offset with broadcasting.
    
    """
    Put back the results into the larger segmentation image. 
    """
    segmentation_pad_clean = np.zeros_like(segmentation_pad)
    min_size = 0
    
    for res_ii in np.arange(len(res)): 
        res_cell = res[res_ii]
        if len(res_cell) > min_size:
            segmentation_pad_clean[res_cell[:,0], 
                                    res_cell[:,1], 
                                    res_cell[:,2]] = cells_id[res_ii]
    
    # unpad and return the final segmentation 
    segmentation_pad_clean = segmentation_pad_clean[padding[0][0]: padding[0][0]+segmentation.shape[0], 
                                                    padding[1][0]: padding[1][0]+segmentation.shape[1],
                                                    padding[2][0]: padding[2][0]+segmentation.shape[2]]
      
    return segmentation_pad_clean
    
     
def guided_filter_3D_cell_segmentation_MP(segmentation,
                                          guide_image,
                                          params):
    
    import skimage.measure as skmeasure
    import skimage.filters as skfilters
    import skimage.morphology as skmorph
    import multiprocess as mp 
    import numpy as np
    import scipy.ndimage as ndimage 
    
    
    n_cpu = params['guide_filter']['n_cpu']
    if n_cpu is None:
        n_cpu = np.maximum(mp.cpu_count() - 1,1)
    
    # guide_img must be in range 0, 1, enforce here. 
    guide_img = usegment3D_filters.normalize(guide_image, 
                                               pmin=params['guide_img']['pmin'], 
                                               pmax=params['guide_img']['pmax'], clip=True)
    
    """
    Do we do additional ridge filtering? and combine based on the meijering filter? 
    """
    if params['ridge_filter']['do_ridge_enhance']:
        if params['ridge_filter']['do_multiprocess_2D']:
            n_cpu_ridge = params['ridge_filter']['n_cpu']
            if n_cpu_ridge is None:
                n_cpu_ridge = np.maximum(mp.cpu_count()//2 - 1,1)
                
            guide_img_edge = usegment3D_filters.meijering_ridge_filter_2D_multiprocess(guide_img, 
                                                                                       sigmas=params['ridge_filter']['sigmas'], 
                                                                                       low_contrast_percentile=params['ridge_filter']['low_contrast_fraction'], 
                                                                                       black_ridges=params['ridge_filter']['black_ridges'], 
                                                                                       n_cpu=n_cpu_ridge)
        else:
            guide_img_edge = skfilters.meijering(guide_img, 
                                                 sigmas=params['ridge_filter']['sigmas'],
                                                 black_ridges=params['ridge_filter']['black_ridges'])
        
        print(guide_img_edge.shape)
        # do the mixing
        guide_img = (1.-params['ridge_filter']['mix_ratio']) * guide_img + (params['ridge_filter']['mix_ratio']) * usegment3D_filters.normalize(guide_img_edge, pmin=params['ridge_filter']['pmin'], pmax=params['ridge_filter']['pmax'], clip=True)
    
    
    """
    define the parameters upfront
    """
    # some setup. 
    pad_size = params['guide_filter']['pad_size']
    size_factor = params['guide_filter']['size_factor'] # this scales the radius of the guided filter by this * mean bounding box diameter, increase to recover longer protrusions.
    
    use_adaptive_radius = params['guide_filter']['adaptive_radius_bool']
    min_protrusion_size = params['guide_filter']['min_protrusion_size'] # set this as the minimum protrusion out.
    mode = params['guide_filter']['mode'] # using additive as cells are empty inside. 
    base_dilate = params['guide_filter']['base_dilate'] # don't dilate
    base_erode = params['guide_filter']['base_erode'] # do erode the original (we can increase this to get better erosions if the input is much fatter), but eroding too much may make the segmentation hollow, as it cannot infill.
    
    guide_filter_eps = params['guide_filter']['eps'] # lower for more effect without changing radius. 
    guide_radius = params['guide_filter']['radius']
    
    threshold_level=params['guide_filter']['threshold_level'] 
    n_thresholds= params['guide_filter']['threshold_nlevels']
    
    use_int = params['guide_filter']['use_int']
    
    """
    Pad the relevant images
    """
    pad = np.hstack([pad_size,pad_size,pad_size])
    padding = ((pad_size,pad_size), (pad_size,pad_size), (pad_size,pad_size))
    guide_img_pad = np.pad(guide_img, padding, mode='reflect')
    segmentation_pad = np.pad(segmentation, padding, mode='constant') # maybe this will be fine!
    
    
    """
    find individual cell ids
    """
    # find out the individual cell ids
    cells = skmeasure.regionprops(segmentation_pad)
    cells_id = np.setdiff1d(np.unique(segmentation_pad), 0 ) # exclude the background label
    
    bboxes = [cells[ii].bbox for ii in range(0, len(cells))]
    
    """
    define the internal function
    """
    def _refine_single_obj_guidedfilter(obj_id):
        
        # bbox = cells[obj_id].bbox
        # area = cells[obj_id].area
        bbox = bboxes[obj_id]
        
        label_bbox = segmentation_pad[bbox[0]-pad[0]:bbox[3]+pad[0],
                                      bbox[1]-pad[1]:bbox[4]+pad[1],
                                      bbox[2]-pad[2]:bbox[5]+pad[2]]
        label_img = guide_img_pad[bbox[0]-pad[0]:bbox[3]+pad[0],
                                  bbox[1]-pad[1]:bbox[4]+pad[1],
                                  bbox[2]-pad[2]:bbox[5]+pad[2]]


        # figure out the mean size. (used if adaptive)        
        mean_size = np.prod(label_bbox.shape)**(1./3) # this seems better. 
        # mean_size = (np.sum((label_bbox==cells_id[obj_id])) * 3/ 4. /(np.pi))**(1./3)
        
        label_binary = label_bbox == cells_id[obj_id]
        
        if use_int:
            label_binary = label_binary * 1
        else:
            label_binary = label_binary * 1. 
        
       
        if use_adaptive_radius: # generally better for many heterogeneous cells. 
            label_bbox_refine_guided = usegment3D_filters.guidedfilter(label_binary, # must be float! 
                                                              label_img, 
                                                               radius=np.maximum(min_protrusion_size, mean_size*size_factor),
                                                              eps=guide_filter_eps)
        # print(np.max(label_img))
        else:
            label_bbox_refine_guided = usegment3D_filters.guidedfilter(label_binary, 
                                                              label_img, 
                                                               radius=guide_radius, # use fixed radius
                                                              eps=guide_filter_eps)
        
        label_bbox_refine = label_bbox_refine_guided>=skfilters.threshold_multiotsu(label_bbox_refine_guided, n_thresholds)[threshold_level]
        
        # get the largest component! 
        label_bbox_refine = usegment3D_filters.largest_component_vol(label_bbox_refine) # takes the largest comp
      
        if mode == 'additive':
            base = (label_bbox==cells_id[obj_id]).copy()
            if base_dilate>0:
                base = skmorph.binary_dilation(base, skmorph.ball(base_dilate))
            elif base_erode>0:
                base = skmorph.binary_erosion(base, skmorph.ball(base_erode))
            else:
                pass
            label_bbox_refine = np.logical_or(label_bbox_refine,
                                              base)
        else:
            # all others do replacement. 
            pass
            
        if params['guide_filter']['collision_erode']>0:
            # # mask with others. # so we don't go over... 
            label_bbox_refine = np.logical_or((label_bbox==cells_id[obj_id]), 
                                              # label_bbox == 0)
                                              skmorph.binary_erosion(label_bbox == 0, skmorph.ball(params['guide_filter']['collision_erode']))) * label_bbox_refine
            # label_bbox_refine = np.logical_or((label_bbox==cells_id[obj_id]), 
            #                                   label_bbox == 0) * label_bbox_refine
            # label_bbox_refine = grad_filters.largest_component_vol(label_bbox_refine) # takes the largest comp
            
        if params['guide_filter']['collision_close']>0:
            label_bbox_refine = skmorph.binary_closing(label_bbox_refine, skmorph.ball(params['guide_filter']['collision_close'])) # generally ok.
        if params['guide_filter']['collision_dilate']>0:
            label_bbox_refine = skmorph.binary_dilation(label_bbox_refine, skmorph.ball(params['guide_filter']['collision_dilate']))
        if params['guide_filter']['collision_fill_holes']>0:
            label_bbox_refine = ndimage.binary_fill_holes(label_bbox_refine)

        label_bbox_refine = usegment3D_filters.largest_component_vol(label_bbox_refine) # takes the largest comp
                    
        label_coords = np.argwhere(label_bbox_refine>0)
        
        if len(label_coords)>0:
            label_coords = label_coords + (bbox[:3]-pad[:3])[None,:] ### this should be correct.... 
        
        # # return old_binary, new_binary, label_coords # just the coords will be good. 
        # print(obj_id)
        return label_coords
    """
    execute the multiprocess
    """
    
    with mp.Pool(n_cpu) as pool: # potentially thread is faster? 
    #     with mp.Pool(n_procs-1) as pool:
    # # with ThreadPool(n_procs) as pool: #586 with multiprocess but dask then does work. 
        res = pool.map(_refine_single_obj_guidedfilter, range(0, len(cells)))
        
    """
    Put back the results into the larger segmentation image. 
    """
    segmentation_pad_clean = np.zeros_like(segmentation_pad)
    min_size = 0
    
    for res_ii in np.arange(len(res)): 
        res_cell = res[res_ii]
        if len(res_cell) > min_size:
            segmentation_pad_clean[res_cell[:,0], 
                                    res_cell[:,1], 
                                    res_cell[:,2]] = cells_id[res_ii]
    
    # unpad and return the final segmentation 
    segmentation_pad_clean = segmentation_pad_clean[padding[0][0]: padding[0][0]+segmentation.shape[0], 
                                                    padding[1][0]: padding[1][0]+segmentation.shape[1],
                                                    padding[2][0]: padding[2][0]+segmentation.shape[2]]
      
    return segmentation_pad_clean, guide_img
    

# this is the multilabel diffusion equivalent for a single binary  ? (is this worth it?)
# def enhance_3D_binary_segmentation():
    
    
    
    
    
    
    
        
        
        
