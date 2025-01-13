#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:55:13 2024

@author: s205272
"""

if __name__=="__main__":
    
    """
    
    This script will demo how to use u-Segment3D direct method with auto-tuned pretrained Cellpose 2D models to segment an unwrapped topography volume of drosophila embryo  
    
    Unlike the other examples, we are going to demo how to use the direct method, applying only to 'xy' and use the direct method cell probability and gradients to segment. 
    
    We will also demonstrate:
        - manually choosing the slice to set diameter
        - use of multiprocessing gradient descent dynamics. 
    
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
    imgfile = '../../example_data/anisotropic/unwrapped_drosophila/topographic_image_I_alt_Suv_opt_t000025.tif'
    
    img = skio.imread(imgfile)
    
    surface_slice_no = 110
    

    # 1. create a save folder to save everything
    savefolder_seg = os.path.join('.', 
                                  'unwrapped_drosophila_segment');
    uSegment3D_fio.mkdir(savefolder_seg)
    

    """
    Visualize the image in max projection, as you can see the cells are very thin and span only a few cells. 
    
    We are not going to resize the image to isotropic but instead proceed to generate 2D segmentations, optimizing the cellpose diameter per slice.  
    """
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Raw')
    plt.imshow(img[surface_slice_no], cmap='gray')
    plt.subplot(132)
    plt.imshow(img[:,img.shape[1]//2], cmap='gray')
    plt.subplot(133)
    plt.imshow(img[:,:,img.shape[2]//2], cmap='gray')
    plt.savefig(os.path.join(savefolder_seg,
                             'input_image_midslice-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    print(img.shape)
    
    # =============================================================================
    #     1. Preprocess the image. 
    # =============================================================================
    
    preprocess_params = uSegment3D_params.get_preprocess_params()
    
    preprocess_params['do_bg_correction'] = True
    preprocess_params['bg_ds']=1
    
    # since the cells are small we are just going to apply upsampling to xy by using a larger voxel_res
    preprocess_params['voxel_res'] = [1,2,2]
        
    img_preprocess = uSegment3D.preprocess_imgs(img, 
                                                preprocess_params)
    img_preprocess = img_preprocess[0]
    print(img_preprocess.shape)
    

    # have a look at the processed. The result should have better contrast and more uniform illumination of the shape. 
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Preprocessed')
    plt.imshow(img_preprocess[surface_slice_no], cmap='gray')
    plt.subplot(132)
    plt.imshow(img_preprocess[:,img_preprocess.shape[1]//2], cmap='gray')
    plt.subplot(133)
    plt.imshow(img_preprocess[:,:,img_preprocess.shape[2]//2], cmap='gray')
    plt.savefig(os.path.join(savefolder_seg,
                             'input_image_midslice-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)

    plt.close('all')

    skio.imsave(os.path.join(savefolder_seg, 'preprocessed_input_image.tif'),
                            np.uint8(255.*img_preprocess))
    # =============================================================================
    # =============================================================================
    # #     2. Run Cellpose using u-Segment3D tuning method with cellpose but only with xy slices
    # =============================================================================
    # =============================================================================
    
    """
    Get some default parameters
    """
    cellpose_segment_params = uSegment3D_params.get_Cellpose_autotune_params()
    cellpose_segment_params['cellpose_modelname'] = 'cyto'
    # cellpose_segment_params['cellpose_modelname'] = 'cyto3'
    cellpose_segment_params['ksize'] = 5 # since cells are small. 
    
    print('========== Cellpose segmentation parameters ========')
    print(cellpose_segment_params)    
    print('============================================')

    """
    Note: the default Cellpose model is set to \'cyto\' which is Cellpose 1., you can change this to any other available Cellpose model e.g. cyto2 is pretty good for single cells.
    """
    cellpose_segment_params['cellpose_modelname'] = 'cyto' # try different models like 'cyto', 'cyto3' for cells, and additionally 'nuclei' for nuclei
    # cellpose_segment_params['cellpose_modelname'] = 'cyto2' # try different models like 'cyto', 'cyto3' for cells, and additionally 'nuclei' for nuclei
    # cellpose_segment_params['cellpose_modelname'] = 'cyto3'
    
    
    # this expects a multichannel input image and in the format (M,N,L,channels) i.e. channels last.
    if len(img_preprocess.shape) == 3:
        img_preprocess = img_preprocess[...,None] # we generate an artificial channel
    
    
    #### 1. running for xy orientation. If the savefolder and basename are specified, the output will be saved as .pkl and .mat files 
    img_segment_2D_xy_diams, img_segment_3D_xy_probs, img_segment_2D_xy_flows, img_segment_2D_xy_styles = uSegment3D.Cellpose2D_model_auto(img_preprocess, 
                                                                                                                                            view='xy', 
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
    aggregation_params['combine_cell_probs']['threshold_n_levels'] = 3
    aggregation_params['combine_cell_probs']['threshold_level'] = 0 # use a lower threshold. 
    
    # add some temporal decay 
    aggregation_params['gradient_descent']['gradient_decay'] = 0.0 # increase this to alleviate splitting of elongated/branching structures with cellpose gradients or decrease towards 0.0 to encourage splitting. 
    aggregation_params['postprocess_binary']['binary_fill_holes'] = True # we want the base segmentations to have holes filled
    
    
    # """
    # a) set this to use the normal aggregation [56 mins / 1h ]
    # """
    # aggregation_params['gradient_descent']['do_mp'] = False # this is default 
    # # probs and gradients should be supplied in the order of [xy, xz, yz]. if one or more do not exist use [] e.g. [xy, [], []]
    # segmentation3D, (probability3D, gradients3D) = uSegment3D.aggregate_2D_to_3D_segmentation_direct_method(probs=[img_segment_3D_xy_probs,
    #                                                                                                                 [],
    #                                                                                                                 []], 
    #                                                                                                         gradients=[img_segment_2D_xy_flows,
    #                                                                                                                     [],
    #                                                                                                                     []], 
    #                                                                                                         params=aggregation_params,
    #                                                                                                         savefolder=None,
    #                                                                                                         basename=None)
    
    """
    b) set this to do multiprocess aggregation [approx 20 mins for 72 threads] 
    """
    aggregation_params['gradient_descent']['do_mp'] = True # this is default 
    aggregation_params['gradient_descent']['tile_shape'] = (len(img_segment_3D_xy_probs), 512, 512) 
    aggregation_params['gradient_descent']['tile_aspect'] = (1, 3, 3)
    aggregation_params['gradient_descent']['tile_overlap_ratio'] = 0.2 # adjust this smaller if cell is small.
    
    # probs and gradients should be supplied in the order of [xy, xz, yz]. if one or more do not exist use [] e.g. [xy, [], []]
    segmentation3D, (probability3D, gradients3D) = uSegment3D.aggregate_2D_to_3D_segmentation_direct_method(probs=[img_segment_3D_xy_probs,
                                                                                                                    [],
                                                                                                                    []], 
                                                                                                            gradients=[img_segment_2D_xy_flows,
                                                                                                                        [],
                                                                                                                        []], 
                                                                                                            params=aggregation_params,
                                                                                                            savefolder=None,
                                                                                                            basename=None)
    
    
    
    # 2. joint the save folder with the filename we wish to use. 2 files will be saved, 1=segmentation as labels and 2nd = 16color RGB colored segmentation for visualization
    uSegment3D_fio.save_segmentation(os.path.join(savefolder_seg,
                                                  'uSegment3D_unwrapped_drosophila_labels.tif'), segmentation3D)
    
    
    # 3. if you want to save the intermediate combined probability and gradients, we recommend using pickle 
    uSegment3D_fio.write_pickle(os.path.join(savefolder_seg,
                                                  'uSegment3D_unwrapped_drosophila_3Dcombined_probs_and_gradients'), 
                                savedict={'probs': probability3D.astype(np.float32),
                                          'gradients': gradients3D.astype(np.float32)})
    
    
    # =============================================================================
    #     3. We can now do postprocessing on size and flow consistency etc. 
    # =============================================================================
    """
    1. first postprocessing we can do is size filtering and flow consistency. 
    """
    postprocess_segment_params = uSegment3D_params.get_postprocess_segmentation_params()
    
    print('========== Default size and flow-consistency filtering parameters ========')
    print(postprocess_segment_params)    
    print('============================================')
    postprocess_segment_params['size_filters']['minsize'] = 15
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
    plt.title('Slices Segmentation Overlay')
    plt.imshow(sksegmentation.mark_boundaries(np.dstack([img_preprocess[surface_slice_no], 
                                                          img_preprocess[surface_slice_no], 
                                                          img_preprocess[surface_slice_no]]),
                                              segmentation3D_filt[surface_slice_no], 
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
    plt.savefig(os.path.join(savefolder_seg,
                             'segmentation_filt_overlay_image_midslices-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')
    
    
    # save this out 
    uSegment3D_fio.save_segmentation(os.path.join(savefolder_seg,
                                                  'uSegment3D_unwrap_drosophila_labels_postprocess.tif'), segmentation3D_filt)

                                   
    
