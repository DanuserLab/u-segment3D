#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:55:13 2024

@author: s205272
"""

if __name__=="__main__":
    
    """
    
    This script will demo how to use u-Segment3D indirect method to perform reconstruction of 3D objects from their 2D slices using the Lateral Root Primordia as example
    This demonstrates how to utilize the indirect method using all three orthoviews. 
    
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
    
    import segment3D.plotting as uSegment3D_plotting
    import segment3D.parameters as uSegment3D_params # this is useful to call default parameters, and keep track of parameter changes and for saving parameters.  
    import segment3D.segmentation as uSegment3D_segment
    import segment3D.usegment3d as uSegment3D
    import segment3D.filters as uSegment3D_filters 
    import segment3D.file_io as uSegment3D_fio
    import segment3D.watershed as uSegment3D_watershed
    import segment3D.flows as uSegment3D_flows
    
    
    # read the lamellipodia image. 
    imgfile = '../../example_data/technical/LateralRootPrimordia/GT_labels.tif'
    
    # load the labels 
    ref_objects = skio.imread(imgfile)
    
    
    """
    Check the background (assumed largest object) has a label = 0 
    """
    if np.min(ref_objects) > 0: 
        uniq_labels = np.unique(ref_objects)
        area_labels = np.hstack([np.sum(ref_objects==lab) for lab in uniq_labels])
        ref_objects[ref_objects == uniq_labels[np.argmax(area_labels)]] = 0 
    
    # color the labels to make it easier to visualize
    ref_objects_color = uSegment3D_plotting.color_segmentation(ref_objects, cmapname='Spectral')
    
    
    """
    Visualize the image midslices. 
    
    I have already resized this image to isotropic resolution and downsized 0.5 x
    """
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Mid_slice')
    plt.imshow(ref_objects_color[ref_objects_color.shape[0]//2])
    plt.subplot(132)
    plt.imshow(ref_objects_color[:,ref_objects_color.shape[1]//2])
    plt.subplot(133)
    plt.imshow(ref_objects_color[:,:,ref_objects_color.shape[2]//2])
    plt.show()
    
   
    
    """
    Generate the 2D slice labels.
    - this transposes the input  volume into xy, xz, yz stacks, then goes slice-by-slice and ensures every ID corresponds to one spatial connected component.
    """
    # xy 
    labels_xy = ref_objects.copy()
    labels_xy = uSegment3D_filters.filter_2d_label_slices(labels_xy,bg_label=0, minsize=8)
    
    # xz 
    labels_xz = ref_objects.transpose(1,0,2).copy()
    labels_xz = uSegment3D_filters.filter_2d_label_slices(labels_xz,bg_label=0, minsize=8)
   
    # xy 
    labels_yz = ref_objects.transpose(2,0,1).copy()
    labels_yz = uSegment3D_filters.filter_2d_label_slices(labels_yz,bg_label=0, minsize=8)
  
    
    
# =============================================================================
#     2. Perform the 2D-to-3D aggregation using all three orthoviews
# =============================================================================
    
    # Get the default parameters. 
    indirect_aggregation_params = uSegment3D_params.get_2D_to_3D_aggregation_params()
    
    # Set the indirect parameters, such as choice of distance transform and adjusting gradient descent parameters. 
    
    # choice of transforms and its computation parameters:
        
    # # e.g. using diffusion centroid
    # indirect_aggregation_params['indirect_method']['dtform_method'] = 'cellpose_improve' # this will use our exact solver of heat equation. with a fixed central point
    # # indirect_aggregation_params['indirect_method']['dtform_method'] = 'fmm' # this will Fast marching. with a fixed central point
    # indirect_aggregation_params['indirect_method']['edt_fixed_point_percentile'] = 0.01 
    # indirect_aggregation_params['gradient_descent']['gradient_decay'] = 0.25 # we used 0.0 for cellpose_improve in the paper, 0.25 seems better.
    
    # # # e.g. using  diffusion skeleton
    indirect_aggregation_params['indirect_method']['dtform_method'] = 'cellpose_skel' # this will use our exact solver of heat equation. with a fixed central point
    # indirect_aggregation_params['indirect_method']['dtform_method'] = 'fmm_skel' # this will use our exact solver of heat equation. with a fixed central point
    indirect_aggregation_params['indirect_method']['smooth_skel_sigma'] = 1 # 1 or 2 is good ? 
    indirect_aggregation_params['gradient_descent']['gradient_decay'] = 0.25 # we used 0.0 for cellpose_improve in the paper
    
    # # e.g. using EDT
    # indirect_aggregation_params['indirect_method']['dtform_method'] = 'edt' # this will use our exact solver of heat equation. with a fixed central point
    # indirect_aggregation_params['gradient_descent']['gradient_decay'] = 0.5 # we used 0.5 for EDT in the paper
    
    
    # set the gradient descent parameters
    indirect_aggregation_params['gradient_descent']['n_iter'] = 250 
    indirect_aggregation_params['gradient_descent']['momenta'] = 0.98 # help boost the splitting 
        
    # we can visualize the dynamics
    indirect_aggregation_params['gradient_descent']['debug_viz'] = True
    

    # perform the indirect method. As we only have xy segmentations, we set the rest to empty []. We need to pass in the shape of the 'xy' view to get the dimensionality correct. 
    segmentation3D, (probability3D, gradients3D) = uSegment3D.aggregate_2D_to_3D_segmentation_indirect_method(segmentations=[labels_xy,
                                                                                                                             labels_xz,
                                                                                                                             labels_yz], 
                                                                                                                  img_xy_shape = ref_objects.shape, 
                                                                                                                precomputed_binary=ref_objects>0, # we know the foreground
                                                                                                                params=indirect_aggregation_params,
                                                                                                                savefolder=None,
                                                                                                                basename=None)
    
    savefolder_seg = './reconstruct_LRP'
    uSegment3D_fio.mkdir(savefolder_seg)
    
    uSegment3D_fio.save_segmentation(os.path.join(savefolder_seg,
                                                  'uSegment3D_reconstuct_LRP_initial_labels.tif'), segmentation3D)
    
    
    # remove small objects
    segmentation3D = uSegment3D_filters.remove_small_labels(segmentation3D,
                                                            min_size=250)
    
    uSegment3D_fio.save_segmentation(os.path.join(savefolder_seg,
                                                  'uSegment3D_reconstuct_LRP_initial_labels_size-filter.tif'), segmentation3D)
    
# =============================================================================
#     3. Measure the AP to see how well we reconstructed.
# =============================================================================
    
    import segment3D.metrics as uSegment3D_metrics
     
    """
    set up the iou thresholds 
    """
    n_samples = 11 # they used 11. 
    iou_thresholds = np.linspace(0.5,1,n_samples)
         
     
    ref_segmentation = ref_objects.copy()
    pred_segmentation = segmentation3D.copy()
     
    """
    relabelling the segmentations by lexsorting on (x,y,z) centroid coordinates. 
    """
    ref_labels_relabel = uSegment3D_segment.reorder_labels_3D(ref_segmentation)
    predict_labels_relabel = uSegment3D_segment.reorder_labels_3D(pred_segmentation)
     
    ap_relabel, tp_relabel, fp_relabel, fn_relabel = uSegment3D_metrics.average_precision([ref_labels_relabel*1], 
                                                                                           [predict_labels_relabel], 
                                                                                           threshold=iou_thresholds)
    ap_curve_relabelled = np.nanmean(ap_relabel, axis=0) # there is only 1 image anyway so we could have taken the first element. 
     
    """
    for comparison: compute the AP without relabelling
    """
    ap, tp, fp, fn = uSegment3D_metrics.average_precision([ref_segmentation*1], 
                                                           [pred_segmentation], 
                                                           threshold=iou_thresholds)
     
    ap_curve_not_relabelled = np.nanmean(ap, axis=0)
     
     
    """
    Coplotting both AP curves.
    """
    import pylab as plt 
     
    plt.figure(figsize=(8,8))
    plt.plot(iou_thresholds,
              ap_curve_relabelled, 'go-', label='with relabelling')
    plt.plot(iou_thresholds, 
              ap_curve_not_relabelled, 'ro-', label='without relabelling')
    plt.legend(loc='best')
    plt.xlim([0.5,1.0])
    plt.ylim([0,1])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Average Precision', fontsize=20)
    plt.xlabel('IoU', fontsize=20)
    plt.show()

     
    

                                   
    