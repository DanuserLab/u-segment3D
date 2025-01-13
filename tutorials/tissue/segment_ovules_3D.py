#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:55:13 2024

@author: s205272
"""

if __name__=="__main__":
    
    """
    
    This script will demo how to use u-Segment3D's auto-tuning with pretrained Cellpose 2D models 
    using direct method and all three orthoviews to generate 3D segmentation for an example image from the Val split of the Ovules dataset. We also show how to compute the AP curve for benchmarking with the provided reference segmentation.
    
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
    import os 
    
    import segment3D.parameters as uSegment3D_params # this is useful to call default parameters, and keep track of parameter changes and for saving parameters.  
    import segment3D.gpu as uSegment3D_gpu
    import segment3D.usegment3d as uSegment3D
    import segment3D.filters as uSegment3D_filters 
    import segment3D.file_io as uSegment3D_fio
    
    
    # Load the already downsampled and isotropic resampled ovules image from the val split. 
    imgfile = '../../example_data/tissue/ovules/raw.tif'
    
    img = skio.imread(imgfile)
    

    # we can save the segmentation with its colors using provided file I/O functions in u-Segment3D. If the savefolder exists and provided with basename in the function above, these would be saved automatically. 
    # 1. create a save folder 
    savecellfolder = os.path.join('.', 
                                  'ovules_segment');
    uSegment3D_fio.mkdir(savecellfolder)
    
    """
    Visualize the image in max projection and mid-slice to get an idea of how it looks
    """
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Max. Projection')
    plt.imshow(img.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(img.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(img.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(savecellfolder, 
                             'input_max-projection_slices.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Mid Slices')
    plt.imshow(img[img.shape[0]//2], cmap='gray')
    plt.subplot(132)
    plt.imshow(img[:,img.shape[1]//2], cmap='gray')
    plt.subplot(133)
    plt.imshow(img[:,:,img.shape[2]//2], cmap='gray')
    plt.savefig(os.path.join(savecellfolder, 
                             'input_midslice-projection_slices.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')
    
    
    # =============================================================================
    #     1. Preprocess the image. 
    # =============================================================================
    
    # generate the default preprocessing parameter settings. 
    preprocess_params = uSegment3D_params.get_preprocess_params()
    
    # this image has been made isotropic, therefore we don't change any of the resizing parameters
    print('========== preprocessing parameters ========')
    print(preprocess_params)    
    print('============================================')
    
    
    """
    Apply gamma correction to boost signal 
    """
    import skimage.exposure as skexposure
    img = skexposure.adjust_gamma(img, gamma=0.5)
    
    
    """
    For this example which is deconvolved we found background illumination correction made it worse.
    """
    preprocess_params['do_bg_correction'] = True
    preprocess_params['bg_ds'] = 1
    
    """
    we will run segmentation at the same scale as the image. 
    """
    preprocess_params['factor'] = 1. # 0.5 wasn't good. 
    # preprocess_params['normalize_min'] = 5.0 # increase the lowest value
    
    
    # run the default preprocessing process in uSegment3D. This process is adapted to multichannel images. therefore for single-channel we need to squeeze the output
    img_preprocess = uSegment3D.preprocess_imgs(img, params=preprocess_params)
    img_preprocess = np.squeeze(img_preprocess)
    
    
    # do median filter
    img_preprocess = ndimage.median_filter(img_preprocess, size=3)
    
    
    # have a look at the processed. The result should have better contrast and more uniform illumination of the shape. 
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Max. Projection of preprocessed')
    plt.imshow(img_preprocess.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(img_preprocess.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(img_preprocess.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(savecellfolder, 
                             'preprocess_max-projection_slices.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Mid Slices of preprocessed')
    plt.imshow(img_preprocess[img_preprocess.shape[0]//2], cmap='gray')
    plt.subplot(132)
    plt.imshow(img_preprocess[:,img_preprocess.shape[1]//2], cmap='gray')
    plt.subplot(133)
    plt.imshow(img_preprocess[:,:,img_preprocess.shape[2]//2], cmap='gray')
    plt.savefig(os.path.join(savecellfolder, 
                             'preprocess_midslice-projection_slices.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')
    
    
    # save the preprocessed image. 
    skio.imsave(os.path.join(savecellfolder, 
                             'preprocessed_image_input.tif'), 
                np.uint8(255.*img_preprocess))

    # =============================================================================
    #     2. Run Cellpose 2D in xy, xz, yz with auto-tuning diameter to get cell probability and gradients, in all 3 views. 
    # =============================================================================
    # we are going to use our automated tuner to run cellpose scanning the stack in xy, xz, and yz to set an automatic diameter for each view
    # this will generate the predicted cell probabilites and gradients
    
    cellpose_segment_params = uSegment3D_params.get_Cellpose_autotune_params()
    
    print('========== Cellpose segmentation parameters ========')
    print(cellpose_segment_params)    
    print('============================================')
    
    
    """
    Note: the default Cellpose model is set to \'cyto\' which is Cellpose 1., you can change this to any other available Cellpose model e.g. cyto2 is pretty good for single cells.
    """
    
    cellpose_segment_params['cellpose_modelname'] = 'cyto' # try different models like 'cyto', 'cyto3' for cells, and additionally 'nuclei' for nuclei
    # cellpose_segment_params['cellpose_modelname'] = 'cyto2' # try different models like 'cyto', 'cyto3' for cells, and additionally 'nuclei' for nuclei
    # cellpose_segment_params['cellpose_modelname'] = 'cyto3'
    
    """
    If the below auto-inferred diameter is picking up noise and being too small, we can increase the default ksize, alternatively we can add median_filter.
    """
    cellpose_segment_params['ksize'] = 21
    
    
    if len(img_preprocess.shape) == 3:
        # this expects a multichannel input image and in the format (M,N,L,channels) i.e. channels last.
        img_preprocess = img_preprocess[...,None] # we generate an artificial channel
        
    
    savefolderplots_xy = os.path.join(savecellfolder, 'cellpose_xy'); uSegment3D_fio.mkdir(savefolderplots_xy)
    cellpose_segment_params['saveplotsfolder'] = savefolderplots_xy
    #### 1. running for xy orientation. If the savefolder and basename are specified, the output will be saved as .pkl and .mat files 
    img_segment_2D_xy_diams, img_segment_3D_xy_probs, img_segment_2D_xy_flows, img_segment_2D_xy_styles = uSegment3D.Cellpose2D_model_auto(img_preprocess, 
                                                                                                                                            view='xy', 
                                                                                                                                            params=cellpose_segment_params, 
                                                                                                                                            basename=None, savefolder=None)

    savefolderplots_xz = os.path.join(savecellfolder, 'cellpose_xz'); uSegment3D_fio.mkdir(savefolderplots_xz)
    cellpose_segment_params['saveplotsfolder'] = savefolderplots_xz
    #### 2. running for xz orientation 
    img_segment_2D_xz_diams, img_segment_3D_xz_probs, img_segment_2D_xz_flows, img_segment_2D_xz_styles = uSegment3D.Cellpose2D_model_auto(img_preprocess, 
                                                                                                                                            view='xz', 
                                                                                                                                            params=cellpose_segment_params, 
                                                                                                                                            basename=None, savefolder=None)

    savefolderplots_yz = os.path.join(savecellfolder, 'cellpose_yz'); uSegment3D_fio.mkdir(savefolderplots_yz)
    cellpose_segment_params['saveplotsfolder'] = savefolderplots_yz
    #### 3. running for yz orientation 
    img_segment_2D_yz_diams, img_segment_3D_yz_probs, img_segment_2D_yz_flows, img_segment_2D_yz_styles = uSegment3D.Cellpose2D_model_auto(img_preprocess, 
                                                                                                                                            view='yz', 
                                                                                                                                            params=cellpose_segment_params, 
                                                                                                                                            basename=None, savefolder=None)

    
    
    # =============================================================================
    #     3. We can now use the predicted probabilies and flows directly to aggregate a 3D consensus segmentation (Direct Method)
    # =============================================================================
    
    aggregation_params = uSegment3D_params.get_2D_to_3D_aggregation_params()
    
    print('========== Default 2D-to-3D aggregation parameters ========')
    print(aggregation_params)    
    print('============================================')
    
    # make sure that we say we are using Cellpose probability predictions which needs to be normalized. If not using Cellpose predicted masks, then set this to be False. We assume this has been appropriately normalized to 0-1
    aggregation_params['combine_cell_probs']['cellpose_prob_mask'] = True 
    
    # add some temporal decay 
    aggregation_params['gradient_descent']['gradient_decay'] = 0.0 # increase this to alleviate splitting of elongated/branching structures with cellpose gradients or decrease towards 0.0 to encourage splitting. 
    aggregation_params['postprocess_binary']['binary_fill_holes'] = True # we want the base segmentations to have holes filled
    
    
    # probs and gradients should be supplied in the order of [xy, xz, yz]. if one or more do not exist use [] e.g. [xy, [], []]
    segmentation3D, (probability3D, gradients3D) = uSegment3D.aggregate_2D_to_3D_segmentation_direct_method(probs=[img_segment_3D_xy_probs,
                                                                                                                    img_segment_3D_xz_probs,
                                                                                                                    img_segment_3D_yz_probs], 
                                                                                                            gradients=[img_segment_2D_xy_flows,
                                                                                                                        img_segment_2D_xz_flows,
                                                                                                                        img_segment_2D_yz_flows], 
                                                                                                            params=aggregation_params,
                                                                                                            savefolder=None,
                                                                                                            basename=None)
    
    
    # 2. joint the save folder with the filename we wish to use. 2 files will be saved, 1=segmentation as labels and 2nd = 16color RGB colored segmentation for visualization
    uSegment3D_fio.save_segmentation(os.path.join(savecellfolder,
                                                  'uSegment3D_ovules_labels.tif'), segmentation3D)
    
    
    # 3. if you want to save the intermediate combined probability and gradients, we recommend using pickle 
    uSegment3D_fio.write_pickle(os.path.join(savecellfolder,
                                                  'uSegment3D_ovules_3Dcombined_probs_and_gradients'), 
                                savedict={'probs': probability3D.astype(np.float32),
                                          'gradients': gradients3D.astype(np.float32)})
    
    
    # =============================================================================
    #     4. We can now do postprocessing 
    # =============================================================================
    """
    1. first postprocessing we can do is size filtering and flow consistency. 
    """
    postprocess_segment_params = uSegment3D_params.get_postprocess_segmentation_params()
    
    print('========== Default size and flow-consistency filtering parameters ========')
    print(postprocess_segment_params)    
    print('============================================')
    
    postprocess_segment_params['flow_consistency']['flow_threshold'] = 0.85 # we can adjust the threshold if needed
    
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
    plt.savefig(os.path.join(savecellfolder,
                             'segmentation_filt_overlay_image_midslices-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')
    
    
    # save this out 
    uSegment3D_fio.save_segmentation(os.path.join(savecellfolder,
                                                  'uSegment3D_ovules_labels_postprocess.tif'), segmentation3D_filt)
    
    
    
    
    # =============================================================================
    # =============================================================================
    # #    This data has provided reference segmentation (see the paper). We can load this in and see how well we are doing.  
    # =============================================================================
    # =============================================================================
    
    ref_segmentfile = '../../example_data/tissue/ovules/GT_labels.tif'
    ref_segmentation = skio.imread(ref_segmentfile)
    
    
    """checking the image size: these two should be the ssame"""
    print('predicted segmentation volume size: ', segmentation3D_filt.shape)
    print('reference segmentation volume size: ', ref_segmentation.shape)
    
    
    """check that if the background is not labelled as 0 then be set as 0"""
    if np.min(ref_segmentation) > 0: 
        uniq_labels = np.unique(ref_segmentation)
        area_labels = np.hstack([np.sum(ref_segmentation==lab) for lab in uniq_labels])
        ref_segmentation[ref_segmentation == uniq_labels[np.argmax(area_labels)]] = 0 
    
    import segment3D.plotting as uSegment3D_plotting
    
    # derive a colored version of the reference for visualisation 
    ref_segmentation_color = uSegment3D_plotting.color_segmentation(ref_segmentation, 
                                                                    cmapname='Spectral', # use the  
                                                                    n_colors=16)
    
    # derive a colored version of the predicted segmentation for visualisation
    pred_segmentation_color = uSegment3D_plotting.color_segmentation(segmentation3D_filt, 
                                                                    cmapname='Spectral', # use the  
                                                                    n_colors=16)
    
    # visualize the midslices.
    plt.figure(figsize=(10,10))
    
    plt.subplot(331)
    plt.title('Mid Slices of preprocessed')
    plt.imshow(img_preprocess[img_preprocess.shape[0]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    plt.subplot(332)
    plt.imshow(img_preprocess[:,img_preprocess.shape[1]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    plt.subplot(333)
    plt.imshow(img_preprocess[:,:,img_preprocess.shape[2]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    
    plt.subplot(334)
    plt.title('Mid Slices of ref. segmentation')
    plt.imshow(ref_segmentation_color[img_preprocess.shape[0]//2]); plt.grid('off'); plt.axis('off')
    plt.subplot(335)
    plt.imshow(ref_segmentation_color[:,img_preprocess.shape[1]//2]); plt.grid('off'); plt.axis('off')
    plt.subplot(336)
    plt.imshow(ref_segmentation_color[:,:,img_preprocess.shape[2]//2]); plt.grid('off'); plt.axis('off')
    
    plt.subplot(337)
    plt.title('Mid Slices of pred. segmentation')
    plt.imshow(pred_segmentation_color[img_preprocess.shape[0]//2]); plt.grid('off'); plt.axis('off')
    plt.subplot(338)
    plt.imshow(pred_segmentation_color[:,img_preprocess.shape[1]//2]); plt.grid('off'); plt.axis('off')
    plt.subplot(339)
    plt.imshow(pred_segmentation_color[:,:,img_preprocess.shape[2]//2]); plt.grid('off'); plt.axis('off')
    plt.savefig(os.path.join(savecellfolder,
                             'final-segmentation_vs_reference_midslices-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')
    
    

    """
    Note: we don't need to ensure evaluation of the same foreground in this case, because the reference is defined everywhere the predicted is. 
    """
    # doing exhaustive optimal matching of predicted and reference segmentation is too slow. We can utilize the expedited matching used by e.g. Cellpose/StarDist but we need to make sure the labelling of the cells are consistently ordered.
    import segment3D.metrics as uSegment3D_metrics
    import segment3D.segmentation as uSegment3D_segment # for the label relabelling
    
    """
    set up the iou thresholds 
    """
    n_samples = 11 # they used 11. 
    iou_thresholds = np.linspace(0.5,1,n_samples)
        
    
    """
    relabelling the segmentations by lexsorting on (x,y,z) centroid coordinates. 
    """
    ref_labels_relabel = uSegment3D_segment.reorder_labels_3D(ref_segmentation)
    predict_labels_relabel = uSegment3D_segment.reorder_labels_3D(segmentation3D_filt)
    
    ap_relabel, tp_relabel, fp_relabel, fn_relabel = uSegment3D_metrics.average_precision([ref_labels_relabel*1], 
                                                                                          [predict_labels_relabel], 
                                                                                          threshold=iou_thresholds)
    
    ap_curve_relabelled = np.nanmean(ap_relabel, axis=0) # there is only 1 image anyway so we could have taken the first element. 
    
    """
    for comparison: compute the AP without relabelling
    """
    ap, tp, fp, fn = uSegment3D_metrics.average_precision([ref_segmentation*1], 
                                                          [segmentation3D_filt], 
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
    plt.savefig(os.path.join(savecellfolder,
                             'ap_curve_final_segmentation.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')

    
    
    # =============================================================================
    # =============================================================================
    # #    Trying to improve: since internally the cell segmentation boundaries is pretty good, we can seek to improve the foreground binary. We can do this using guided filter. 
    # =============================================================================
    # =============================================================================
    
    
    """
    Apply guided filter to the foreground binary (i.e. segmentation>0, all foreground regions), using the original image as the 'guide image'. 
    You can see the guided filter feathers the segmentation - we re-threshold this to derive a new foreground binary which we use to mask the segmentation. 
    """
    fg_binary = segmentation3D_filt >0 # derive the current
    
    # use the original image with normalized intensity as guide, with gamma correction 
    guide = uSegment3D_filters.normalize(skexposure.adjust_gamma(img, gamma=0.5),
                                         pmin=2, pmax=99.8, clip=True)# intensities must be 0-1
    guide = ndimage.median_filter(guide, size=5) # remove salt and pepper
    
    
    # we apply guided filter on the binary as a floating image. 
    fg_binary_filter = uSegment3D_filters.guidedfilter(fg_binary*1., 
                                                       G=guide, 
                                                       radius=25, # the radius dictates the spatial extent of correction. Increase to try to correct a larger area. 
                                                       eps=1e-4)
    
    import skimage.filters as skfilters
    import skimage.morphology as skmorph
    
    # standard 2-class Otsu
    fg_binary_filter_threshold = fg_binary_filter >= skfilters.threshold_multiotsu(fg_binary_filter,2)[-1]
    # fg_binary_filter_threshold = uSegment3D_filters.largest_component_vol(fg_binary_filter_threshold)
    fg_binary_filter_threshold = skmorph.binary_closing(fg_binary_filter_threshold, skmorph.ball(1))
    fg_binary_filter_threshold = ndimage.binary_fill_holes(fg_binary_filter_threshold)
    
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(331)
    plt.title('Mid Slices of pred. segmentation foreground')
    plt.imshow(fg_binary[fg_binary.shape[0]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    plt.subplot(332)
    plt.imshow(fg_binary[:,fg_binary.shape[1]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    plt.subplot(333)
    plt.imshow(fg_binary[:,:,fg_binary.shape[2]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    
    plt.subplot(334)
    plt.title('Mid Slices of guided filtered pred. segmentation foreground')
    plt.imshow(fg_binary_filter[fg_binary.shape[0]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    plt.subplot(335)
    plt.imshow(fg_binary_filter[:,fg_binary.shape[1]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    plt.subplot(336)
    plt.imshow(fg_binary_filter[:,:,fg_binary.shape[2]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    
    
    plt.subplot(337)
    plt.title('Mid Slices of guided filtered pred. segmentation foreground rebinarize')
    plt.imshow(fg_binary_filter_threshold[fg_binary.shape[0]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    plt.subplot(338)
    plt.imshow(fg_binary_filter_threshold[:,fg_binary.shape[1]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    plt.subplot(339)
    plt.imshow(fg_binary_filter_threshold[:,:,fg_binary.shape[2]//2], cmap='gray'); plt.grid('off'); plt.axis('off')
    plt.savefig(os.path.join(savecellfolder,
                             'final_segmentation_with_guided-filter_binary_viz.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')
    
    
    
    """
    Mask with the original segmented cells and recompute AP. 
    """
    predict_labels_relabel_mask = predict_labels_relabel*fg_binary_filter_threshold
    # measure volumes and remove cells that are too small. 
    predict_labels_relabel_mask = uSegment3D_filters.remove_small_labels(predict_labels_relabel_mask, min_size=250)
    
    ap_relabel_mask, tp_relabel_mask, fp_relabel_mask, fn_relabel_mask = uSegment3D_metrics.average_precision([ref_labels_relabel*1], 
                                                                                                              [predict_labels_relabel_mask], 
                                                                                                              threshold=iou_thresholds)
    
    ap_curve_relabelled_mask = np.nanmean(ap_relabel_mask, axis=0) # there is only 1 image anyway so we could have taken the first element. 
    
    
    plt.figure(figsize=(8,8))
    plt.plot(iou_thresholds,
             ap_curve_relabelled, 'go-', label='with relabelling')
    plt.plot(iou_thresholds, 
             ap_curve_not_relabelled, 'ro-', label='without relabelling')
    plt.plot(iou_thresholds,
             ap_curve_relabelled_mask, 'ko-', label='with relabelling and mask')
    plt.legend(loc='best')
    plt.xlim([0.5,1.0])
    plt.ylim([0,1])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Average Precision', fontsize=20)
    plt.xlabel('IoU', fontsize=20)
    plt.savefig(os.path.join(savecellfolder,
                             'ap_curve_final_segmentation_and_guidedfilter-binary.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')
    
    
    """
    We note the performance is pretty similar with or without the filter, showing the segmentation is pretty good.  
    """
    
    
    
    
    
    
    
    
    
    
