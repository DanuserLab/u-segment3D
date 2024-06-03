#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:55:13 2024

@author: s205272
"""

if __name__=="__main__":
    
    """
    
    This script will demo how to use u-Segment3D's methods to auto-tune pretrained Cellpose 2D models and parse 2D segmentations, then use indirect method to aggregate them together into a single 3D segmentation from one orthogonal view (xy)
    
    We assume example data is downloaded and unzipped into the example_data/ folder located in the root folder of the Github Repository, otherwise please change the filepaths accordingly. 
    
    """
    
    # =============================================================================
    #     0. Read the image data
    # =============================================================================
    
    import skimage.io as skio 
    import pylab as plt 
    import numpy as np 
    import scipy.ndimage as ndimage
    import skimage.segmentation as sksegmentation 
    import skimage.filters as skfilters
    import os 
    
    import segment3D.parameters as uSegment3D_params # this is useful to call default parameters, and keep track of parameter changes and for saving parameters.  
    import segment3D.segmentation as uSegment3D_segment
    import segment3D.usegment3d as uSegment3D
    import segment3D.filters as uSegment3D_filters 
    import segment3D.file_io as uSegment3D_fio
    import segment3D.watershed as uSegment3D_watershed
    import segment3D.flows as uSegment3D_flows
    
    
    # read the lamellipodia image. 
    imgfile = '../../example_data/anisotropic/single-cell-tracking-challenge/t000.tif'
    
    img = skio.imread(imgfile)
    
    
    """
    Visualize the image in max projection, as you can see the cells are very thin and span only a few cells. 
    
    We are not going to resize the image to isotropic but instead proceed to generate 2D segmentations, optimizing the cellpose diameter per slice.  
    """
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Max. Projection')
    plt.imshow(img.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(img.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(img.max(axis=2), cmap='gray')
    plt.show()
    
    
    
    # =============================================================================
    #     1. Preprocess the image. 
    # =============================================================================
    
    # we run a median filter and normalize the image intensity 
    img_preprocess = uSegment3D_filters.normalize(img, clip=True)
    
    
    # have a look at the processed. The result should have better contrast and more uniform illumination of the shape. 
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Max. Projection of preprocessed')
    plt.imshow(img_preprocess.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(img_preprocess.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(img_preprocess.max(axis=2), cmap='gray')
    plt.show()
    
    
    
    # =============================================================================
    # =============================================================================
    # #     2. Run Cellpose 2D slice-by-slice, optimizing the parameter and parsing the segmentation using u-Segment3D method
    # =============================================================================
    # =============================================================================
    
    """
    Get some default parameters
    """
    cellpose_segment_params = uSegment3D_params.get_Cellpose_autotune_params()
    cellpose_segment_params['cellpose_modelname'] = 'cyto'
    cellpose_segment_params['ksize'] = 21
    
    print('========== Cellpose segmentation parameters ========')
    print(cellpose_segment_params)    
    print('============================================')
    
    
    # also parameters for gradient descent 
    aggregation_params = uSegment3D_params.get_2D_to_3D_aggregation_params()
    
    aggregation_params['gradient_descent']['n_iter'] = 100 
    aggregation_params['gradient_descent']['gradient_decay'] = 0.01 # increase this to alleviate splitting of elongated/branching structures with cellpose gradients or decrease towards 0.0 to encourage splitting. 
    aggregation_params['gradient_descent']['momenta'] = 0.98 # help boost the splitting 
    aggregation_params['combine_cell_probs']['threshold_n_levels'] = 3
    aggregation_params['combine_cell_probs']['threshold_level'] = 0
    aggregation_params['combine_cell_probs']['min_prob_thresh'] = 0.1
    
    
    # also postprocessing parameters
    postprocess_segment_params = uSegment3D_params.get_postprocess_segmentation_params()
    
    print('========== Default size and flow-consistency filtering parameters ========')
    print(postprocess_segment_params)    
    print('============================================')
    
    postprocess_segment_params['size_filters']['min_size'] = 15 # minimum 2D cell size
    postprocess_segment_params['flow_consistency']['flow_threshold'] = 0.85 # we can adjust the threshold if needed
    
    
    # initialize lists to store params. 
    all_probs_stack = []
    all_flows_stack = []
    all_masks_stack = []
    all_styles_stack = []
    
    all_params_diams_range = []
    all_params_diams_score = []
    all_params_diams = []
    
    
    """
    Define Cellpose model and it
    """
    from cellpose import models
    import cellpose.utils
    
    if cellpose_segment_params['cellpose_channels'] == 'grayscale':
        cellpose_channels = [0,0] # this is required for running Cellpose in grayscale mode
    else:
        cellpose_channels = [2,1] # green- cyto, red- nuclei
    model = models.Cellpose(model_type=cellpose_segment_params['cellpose_modelname'], 
                            gpu=cellpose_segment_params['gpu'])
    
    
    # iterate over the stack
    for dd in np.arange(len(img_preprocess))[:]:
        
        im_slice = img_preprocess[dd].copy()
        im_slice = ndimage.median_filter(im_slice, size=3) # do a small median filter to suppress noise. 
        
        params, (all_probs, all_flows, all_styles) = uSegment3D_segment.apply_cellpose_model_2D_prob_slice(im_slice,
                                                                                                            model,
                                                                                                            model_channels=cellpose_channels,
                                                                                                            best_diam=cellpose_segment_params['best_diam'],
                                                                                                            model_invert=cellpose_segment_params['model_invert'],
                                                                                                            test_slice=cellpose_segment_params['test_slice'],
                                                                                                            diam_range=cellpose_segment_params['diam_range'],
                                                                                                            ksize=cellpose_segment_params['ksize'],
                                                                                                            smoothwinsize=cellpose_segment_params['smoothwinsize'],
                                                                                                            hist_norm=cellpose_segment_params['hist_norm'],
                                                                                                            kernel_size=cellpose_segment_params['histnorm_kernel_size'],
                                                                                                            clip_limit=cellpose_segment_params['histnorm_clip_limit'],
                                                                                                            use_edge=cellpose_segment_params['use_edge'],
                                                                                                            use_prob_weighted_score=cellpose_segment_params['use_prob_weighted_score'])
            
        all_params_diams_range.append(params[0])
        all_params_diams_score.append(params[1])
        all_params_diams.append(params[2])
        
        ##### the main 
        all_probs = all_probs[0]
        all_flows = all_flows[:,0]
        all_styles = all_styles
            
        all_probs = np.clip(all_probs, -88.72, 88.72)
        all_probs = (1./(1.+np.exp(-all_probs))) # normalize. 
        
        all_flows = np.array([ndimage.gaussian_filter(all_flows[ch_no], sigma=1) for ch_no in np.arange(len(all_flows))])
        all_flows = all_flows/(np.linalg.norm(all_flows, axis=0)[None,...] + 1e-20)
        
        prob_thresh = np.maximum(int(skfilters.threshold_multiotsu(all_probs, 
                                                                           aggregation_params['combine_cell_probs']['threshold_n_levels'])[aggregation_params['combine_cell_probs']['threshold_level']]*10)/10., 
                                         aggregation_params['combine_cell_probs']['min_prob_thresh']) 
        binary = all_probs >= prob_thresh
        cell_seg_connected_original, cell_seg_connected, tracks, votes_grid_acc = uSegment3D_watershed.gradient_watershed2D_binary(binary, 
                                                                                                                              gradient_img=all_flows.transpose(1,2,0),
                                                                                                                              momenta=aggregation_params['gradient_descent']['momenta'],
                                                                                                                              gradient_decay=aggregation_params['gradient_descent']['gradient_decay'],
                                                                                                                              n_iter=aggregation_params['gradient_descent']['n_iter'])
                                                                                    
        """
        remove bad flow masks
        """
        cell_seg_connected_original, _, _ = uSegment3D_flows.remove_bad_flow_masks_2D(cell_seg_connected_original, 
                                                                                      flow=all_flows, 
                                                                                      flow_threshold=postprocess_segment_params['flow_consistency']['flow_threshold'], # increase this ?
                                                                                      dtform_method=postprocess_segment_params['flow_consistency']['dtform_method'],  
                                                                                      fixed_point_percentile=postprocess_segment_params['flow_consistency']['edt_fixed_point_percentile'], 
                                                                                      n_processes=postprocess_segment_params['flow_consistency']['n_cpu'],
                                                                                       power_dist=postprocess_segment_params['flow_consistency']['power_dist'],
                                                                                      alpha=aggregation_params['combine_cell_gradients']['alpha'], 
                                                                                      filter_scale = aggregation_params['combine_cell_gradients']['ksize'])
        
        """
        remove too small areas. 
        """
        cell_seg_connected_original = cellpose.utils.fill_holes_and_remove_small_masks(cell_seg_connected_original, 
                                                                                        min_size=postprocess_segment_params['size_filters']['min_size']) ### this should be implemented to not depend on cellpose!. 
        # cell_seg_connected_original = grad_filters.largest_component_vol_labels_fast(cell_seg_connected_original, connectivity=1)
        
        """
        Overlay the segmentation to visualizing see the segmentation
        """
        marked = sksegmentation.mark_boundaries(np.dstack([im_slice, im_slice, im_slice]), 
                                                cell_seg_connected_original, mode='thick')
        plt.figure()
        plt.imshow(marked)                                         
        plt.show()
        
        
        all_probs_stack.append(all_probs)
        all_flows_stack.append(all_flows)
        all_masks_stack.append(cell_seg_connected_original)
        all_styles_stack.append(all_styles)
        

    all_probs_stack = np.array(all_probs_stack)
    all_flows_stack = np.array(all_flows_stack)
    all_masks_stack = np.array(all_masks_stack)
    all_styles_stack = np.array(all_styles_stack)
        
    
    all_params_diams_range = np.array(all_params_diams_range)
    all_params_diams_score = np.array(all_params_diams_score)
    all_params_diams = np.array(all_params_diams)
    
        
        
    
    # """
    # Save out all the intermediates  (check out zarr for lazy loading these as metadata in one file. )
    # """
    
    import scipy.io as spio
    
    # create a savefolder:
    savefolder_seg = os.path.join('.', 
                                  'single-cell-tracking-challenge_segment')
    uSegment3D_fio.mkdir(savefolder_seg)
    
    view = 'xy'
    basename = os.path.split(imgfile)[-1].split('.tif')[0]
    
    spio.savemat(os.path.join(savefolder_seg,
                                basename+'_cp_flows_%s.mat' %(str(view))),
                              {'flow': all_flows_stack.astype(np.float32)})
                              
    spio.savemat(os.path.join(savefolder_seg,
                                basename+'_cp_probs_%s.mat' %(str(view))),
                              {'prob': all_probs_stack.astype(np.float32)})
    
    skio.imsave(os.path.join(savefolder_seg,
                                basename+'_cp_labels_%s.tif' %(str(view))),
                np.uint16(all_masks_stack))
    
    spio.savemat(os.path.join(savefolder_seg,
                              basename+'_cp_params_%s.mat' %(str(view))),
                  {'style': all_styles_stack,
                  'diam_range':all_params_diams_range,
                  'diam_score':all_params_diams_score,
                  'best_diam':all_params_diams})
    
    
# =============================================================================
#     2. Postfiltering by using mean intensity 
# =============================================================================
    
    # use the normalized as the reference
    ref_im = img_preprocess.copy()
    
    # use a new variable name
    labels_xy = all_masks_stack.copy()
    
    # ### measure the intensities
    mean_intensities = [ndimage.mean(ref_im, labels=labels_xy_slice, index=np.setdiff1d(np.unique(labels_xy_slice),0)) for labels_xy_slice in labels_xy]
    mean_intensities_array = np.hstack(mean_intensities)
    
    # use Otsu to find the cutoff.
    ref_im_thresh = skfilters.threshold_otsu(mean_intensities_array)
    
    
    labels_xy_refine = []
    # better to write a loop. 
    for zz in np.arange(len(labels_xy)):
        
        labels_xy_slice = labels_xy[zz].copy()
        labels_xy_slice_new = labels_xy_slice.copy()
        # measure the mean intensity 
        labs = np.setdiff1d(np.unique(labels_xy_slice), 0)
        
        for lab in labs:
            intensity_lab = np.mean(ref_im[zz][labels_xy_slice==lab])
            if intensity_lab <= ref_im_thresh:
                labels_xy_slice_new[labels_xy_slice==lab] = 0
                
        labels_xy_refine.append(labels_xy_slice_new)
    
    labels_xy_refine = np.array(labels_xy_refine)
    
    
# =============================================================================
#     3. Perform the 2D-to-3D aggregation using only this xy view segmentation. 
# =============================================================================
    
    # Get the default parameters. 
    indirect_aggregation_params = uSegment3D_params.get_2D_to_3D_aggregation_params()
    
    # Set the indirect parameters, such as choice of distance transform and adjusting gradient descent parameters. 
    
    # since we are going to combine on binaries 
    indirect_aggregation_params['indirect_method']['smooth_sigma'] = 1 # we can add some smoothing filter first. 
    
    indirect_aggregation_params['combine_cell_probs']['ksize'] = 1 
    indirect_aggregation_params['combine_cell_probs']['alpha'] = 0.5 
    indirect_aggregation_params['combine_cell_probs']['smooth_sigma'] = 1 # add post smoothing too after combine 
    
    # choice of transforms and its computation parameters:
    indirect_aggregation_params['indirect_method']['dtform_method'] = 'cellpose_improve' # this will use our exact solver of heat equation. with a fixed central point
    indirect_aggregation_params['indirect_method']['edt_fixed_point_percentile'] = 0.01 # adjust the medial centroid placement
    indirect_aggregation_params['indirect_method']['edt_fixed_point_percentile'] = 0.01 

    
    indirect_aggregation_params['gradient_descent']['n_iter'] = 200 
    indirect_aggregation_params['gradient_descent']['gradient_decay'] = 0.05 # increase this to alleviate splitting of elongated/branching structures with cellpose gradients or decrease towards 0.0 to encourage splitting. 
    indirect_aggregation_params['gradient_descent']['momenta'] = 0.98 # help boost the splitting 
    indirect_aggregation_params['combine_cell_probs']['threshold_n_levels'] = 2
    indirect_aggregation_params['combine_cell_probs']['threshold_level'] = -1
    indirect_aggregation_params['combine_cell_probs']['min_prob_thresh'] = 0.1
        

    # perform the indirect method. As we only have xy segmentations, we set the rest to empty []. We need to pass in the shape of the 'xy' view to get the dimensionality correct. 
    segmentation3D, (probability3D, gradients3D) = uSegment3D.aggregate_2D_to_3D_segmentation_indirect_method(segmentations=[labels_xy_refine,
                                                                                                                            [],
                                                                                                                            []], 
                                                                                                                  img_xy_shape = img_preprocess.shape, 
                                                                                                                precomputed_binary=None,
                                                                                                                params=indirect_aggregation_params,
                                                                                                                savefolder=None,
                                                                                                                basename=None)
    
    
    
    
    
    uSegment3D_fio.save_segmentation(os.path.join(savefolder_seg,
                                                  'uSegment3D_single-cell-tracking_initial_labels.tif'), segmentation3D)
    
    
    # =============================================================================
    #     4. Now do postprocessing removing size. 
    # =============================================================================
    """
    1. first postprocessing we can do is size filtering. 
    """
    postprocess_segment_params = uSegment3D_params.get_postprocess_segmentation_params()
    
    print('========== Default size and flow-consistency filtering parameters ========')
    print(postprocess_segment_params)    
    print('============================================')
    
    postprocess_segment_params['size_filters']['min_size'] = 200 
    postprocess_segment_params['size_filters']['do_stats_filter'] = True
    postprocess_segment_params['do_flow_remove'] = True # this currently assumes 3 directions are defined, therefore technically we don't need to set this. 
    
    segmentation3D_filt, flow_consistency_intermediates = uSegment3D.postprocess_3D_cell_segmentation(segmentation3D,
                                                                                                      aggregation_params=aggregation_params,
                                                                                                      postprocess_params=postprocess_segment_params,
                                                                                                      cell_gradients=gradients3D,
                                                                                                      savefolder=None,
                                                                                                      basename=None)
                                              
    
    ### we can overlay the midslice segmentation to the midslices of the image to check segmentation. You can see its quite good, but doesn't capture subcellular details
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Mid Slices Segmentation Overlay')
    plt.imshow(sksegmentation.mark_boundaries(np.dstack([img_preprocess[img_preprocess.shape[0]//2], 
                                                          img_preprocess[img_preprocess.shape[0]//2], 
                                                          img_preprocess[img_preprocess.shape[0]//2]]),
                                              segmentation3D_filt[img_preprocess.shape[0]//2], 
                                              color=(0,1,0), mode='thick'))
    plt.subplot(132)
    plt.imshow(sksegmentation.mark_boundaries(np.dstack([img_preprocess[:,img_preprocess.shape[1]//2], 
                                                          img_preprocess[:,img_preprocess.shape[1]//2], 
                                                          img_preprocess[:,img_preprocess.shape[1]//2]]),
                                              segmentation3D_filt[:,img_preprocess.shape[1]//2], 
                                              color=(0,1,0), mode='thick'))
    plt.subplot(133)
    plt.imshow(sksegmentation.mark_boundaries(np.dstack([img_preprocess[:,:,img_preprocess.shape[2]//2], 
                                                          img_preprocess[:,:,img_preprocess.shape[2]//2], 
                                                          img_preprocess[:,:,img_preprocess.shape[2]//2]]),
                                              segmentation3D_filt[:,:,img_preprocess.shape[2]//2], 
                                              color=(0,1,0), mode='thick'))
    plt.show()
    
    
    # save the segmentation 
    uSegment3D_fio.save_segmentation(os.path.join(savefolder_seg,
                                basename+'_cp_labels_%s_3D_consensus.tif' %(str(view))),
                                     np.uint16(segmentation3D_filt))
    
# =============================================================================
# =============================================================================
# #     Improving the adherence to the cell edges. The segmentation is a bit 'loose'. What we can do fairly simply is to do a binary threshold in 3D as a foreground mask and exclude all cells outside of this. This helps better exclude v. low intensity segmentations
# =============================================================================
# =============================================================================

    import skimage.morphology as skmorph 
    import skimage.color as skcolor 
    import seaborn as sns 
    
    # slight smoothing. 
    img_preprocess_smooth = ndimage.gaussian_filter(img_preprocess, sigma=1)
    mask3D = img_preprocess_smooth >= skfilters.threshold_multiotsu(img_preprocess_smooth, 3)[-1]
    mask3D = skmorph.binary_closing(mask3D, skmorph.ball(3))
    
    final_mask = mask3D*segmentation3D_filt
    final_mask = uSegment3D_filters.largest_component_vol_labels_fast(final_mask) # ensure each label is a single-component
    
    
    final_mask_color = np.uint8(255*skcolor.label2rgb(final_mask, 
                                                  colors=sns.color_palette('Spectral', n_colors=16),
                                                  bg_label=0))
    
    plt.figure(figsize=(10,10))
    plt.imshow(final_mask_color[20])
    plt.show()
        
    
    uSegment3D_fio.save_segmentation(os.path.join(savefolder_seg,
                                basename+'_binary_3D_Otsu_Foreground_mask.tif'),
                                     np.uint16(mask3D))
    
    # save the final segmentation, masked with binary 
    uSegment3D_fio.save_segmentation(os.path.join(savefolder_seg,
                            basename+'_cp_labels_%s_3D_consensus_with_Foreground_mask.tif' %(str(view))),
                                 np.uint16(final_mask))





                                   
    