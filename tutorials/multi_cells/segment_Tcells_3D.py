#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:55:13 2024

@author: s205272
"""

if __name__=="__main__":
    
    """
    
    This script will demo how to use u-Segment3D's auto-tuning with pretrained Cellpose 2D models 
    to generate 3D segmentations for single cells in co-culture where they may be touching neighbor cells and where the cell internal is only weakly stained i.e. is hollow.
    
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
    
    
    # read the lamellipodia image. 
    imgfile = '../../example_data/multi_cells/Tcells/Frame_00000.tif'
    
    img = skio.imread(imgfile)
    
    
    # we can save the segmentation with its colors using provided file I/O functions in u-Segment3D. If the savefolder exists and provided with basename in the function above, these would be saved automatically. 
    # 1. create a save folder 
    savecellfolder = os.path.join('.', 
                                  'Tcells_segment');
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
                             'input_image_max-projection.png'), dpi=300, bbox_inches='tight')
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
                             'input_image_midslices-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    # =============================================================================
    #     1. Preprocess the image, so that it is isotropic
    # =============================================================================
    
    # generate the default preprocessing parameter settings. 
    preprocess_params = uSegment3D_params.get_preprocess_params()
    
    # this image is isotropic, therefore we don't change any of the parameters
    print('========== preprocessing parameters ========')
    print(preprocess_params)    
    print('============================================')
    
    """
    1. This image is isotropic. 
    """
    preprocess_params['voxel_res'] = [1., 1., 1.] # set this as the voxel size. 
    
    """
    For this example which is deconvolved we found background illumination correction made it worse.
    """
    preprocess_params['do_bg_correction'] = True
    
    """
    we will run segmentation at a downsampled size as this will yield diameters better aligning with cellpose 
    """
    preprocess_params['factor'] = 0.5 
    
    """
    Some cells are close to edge. We are going to pad 
    """
    pad_size = 50
    img = np.pad(img, [[pad_size, pad_size], [pad_size, pad_size], [pad_size, pad_size]], mode='constant')
    # infill zero values which may give discontinuity errors. 
    img[img==0] = np.nanmedian(img[img>0])
    
    img = ndimage.median_filter(img, size=3) # remove some of the noise 
    
    
    # run the default preprocessing process in uSegment3D. This process is adapted to multichannel images. therefore for single-channel we need to squeeze the output
    img_preprocess = uSegment3D.preprocess_imgs(img, params=preprocess_params)
    img_preprocess = np.squeeze(img_preprocess)
    
    
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
                             'preprocessed_image_max-projection.png'), dpi=300, bbox_inches='tight')
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
                             'preprocessed_image_midslices-projection.png'), dpi=300, bbox_inches='tight')
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
    # cellpose_segment_params['cellpose_modelname'] = 'cyto3' # seems to break more fragments. 
    
    
    """
    Since the inside is hollow, invert the model (as of Cellpose3, this seems not needed, and makes segmentation worse...)
    """
    cellpose_segment_params['model_invert'] = False
    
    
    """
    If the below auto-inferred diameter is picking up noise and being too small, we can increase the default ksize, alternatively we can add median_filter.
    """
    cellpose_segment_params['ksize'] = 15
    
    # histogram equalize for cell segment too 
    cellpose_segment_params['hist_norm'] = True
    cellpose_segment_params['histnorm_clip_limit'] = 0.01
    cellpose_segment_params['histnorm_kernel_size'] = (512,512)
    cellpose_segment_params['use_edge'] = False # using the brightest signal 
    cellpose_segment_params['diam_range'] = np.arange(5,101,2.5)
    
    
    # this expects a multichannel input image and in the format (M,N,L,channels) i.e. channels last.
    if len(img_preprocess.shape) ==3: # 3D volume 
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
    aggregation_params['combine_cell_probs']['threshold_n_levels'] = 3

    # add some temporal decay 
    aggregation_params['gradient_descent']['gradient_decay'] = 0.25 # increase this to alleviate splitting of elongated/branching structures with cellpose gradients or decrease towards 0.0 to encourage splitting. 
    # aggregation_params['postprocess_binary']['binary_fill_holes'] = True # we want the base segmentations to have holes filled
    
    
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
                                                  'uSegment3D_Tcells_labels.tif'), segmentation3D)
    
    
    # 3. if you want to save the intermediate combined probability and gradients, we recommend using pickle 
    uSegment3D_fio.write_pickle(os.path.join(savecellfolder,
                                                  'uSegment3D_Tcells_3Dcombined_probs_and_gradients'), 
                                savedict={'probs': probability3D.astype(np.float32),
                                          'gradients': gradients3D.astype(np.float32)})
    
    
    # =============================================================================
    #     4. We can now do postprocessing 
    # =============================================================================
    """
    1. first postprocessing we can do is size filtering. 
    """
    postprocess_segment_params = uSegment3D_params.get_postprocess_segmentation_params()
    
    print('========== Default size and flow-consistency filtering parameters ========')
    print(postprocess_segment_params)    
    print('============================================')
    
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
    
    
    """
    2. second postprocessing is to enhance the segmentation. 
        a) diffusing the labels to improve concordance with image boundaries 
        b) guided filter to recover detail such as subcellular protrusions on the surface of the cell. 
    """
    
    ###### a) label diffusion.
    label_diffusion_params = uSegment3D_params.get_label_diffusion_params()
    label_diffusion_params['diffusion']['refine_iters'] = 25 
    label_diffusion_params['diffusion']['refine_alpha'] = 0.5 
    label_diffusion_params['diffusion']['refine_clamp'] = 0.75
    
    
    # customize the diffusion image. 
    guide_image = img_preprocess[...,0].copy()
    
    import skimage.exposure as skexposure
    
    guide_image = ndimage.gaussian_filter(guide_image, sigma=1) 
    guide_image = skexposure.adjust_gamma(guide_image, gamma=0.4) # increase the contrast.  
    guide_image = uSegment3D_filters.normalize(guide_image, clip=True)
    
    
    """
    Remove cells without enough intensity 
    """
    # use the raw.
    guide_img_intensity = uSegment3D_filters.normalize(img_preprocess[...,0], clip=True)
    
    cells_id = np.setdiff1d(np.unique(segmentation3D_filt), 0 )
    
    remove_ids = []
    for rr in cells_id:
        mean_I = np.mean(guide_img_intensity[segmentation3D_filt==rr])
       
        if mean_I < np.mean(guide_img_intensity):
            remove_ids.append(rr)
        
    # delete these!. 
    for rr in remove_ids:
        segmentation3D_filt[segmentation3D_filt == rr] = 0 
    
    
    """
    now run the diffusion 
    """
    segmentation3D_filt_diffuse = uSegment3D.label_diffuse_3D_cell_segmentation_MP(segmentation3D_filt,
                                                                                   guide_image = img_preprocess[...,0],
                                                                                   params=label_diffusion_params)
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Mid Slices Segmentation Overlay (Post label diffusion)')
    plt.imshow(sksegmentation.mark_boundaries(np.dstack([img_preprocess[img_preprocess.shape[0]//2], 
                                                          img_preprocess[img_preprocess.shape[0]//2], 
                                                          img_preprocess[img_preprocess.shape[0]//2]]),
                                              segmentation3D_filt_diffuse[img_preprocess.shape[0]//2], 
                                              color=(0,1,0), mode='thick'))
    plt.subplot(132)
    plt.imshow(sksegmentation.mark_boundaries(np.dstack([img_preprocess[:,img_preprocess.shape[1]//2], 
                                                          img_preprocess[:,img_preprocess.shape[1]//2], 
                                                          img_preprocess[:,img_preprocess.shape[1]//2]]),
                                              segmentation3D_filt_diffuse[:,img_preprocess.shape[1]//2], 
                                              color=(0,1,0), mode='thick'))
    plt.subplot(133)
    plt.imshow(sksegmentation.mark_boundaries(np.dstack([img_preprocess[:,:,img_preprocess.shape[2]//2], 
                                                          img_preprocess[:,:,img_preprocess.shape[2]//2], 
                                                          img_preprocess[:,:,img_preprocess.shape[2]//2]]),
                                              segmentation3D_filt_diffuse[:,:,img_preprocess.shape[2]//2], 
                                              color=(0,1,0), mode='thick'))
    plt.savefig(os.path.join(savecellfolder,
                             'segmentation_label-diffuse_overlay_image_midslices-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')
    
    
    # save this segmentation
    uSegment3D_fio.save_segmentation(os.path.join(savecellfolder,
                                                  'uSegment3D_Tcells_labels_postprocess-diffuse.tif'), segmentation3D_filt_diffuse)
    
    
    
    ##### b) guided filter, (but we are going to do this at original resolution)
    guided_filter_params = uSegment3D_params.get_guided_filter_params()
    
    #  for guided image we will use the input to Cellpose and specify we want to further do ridge filter enhancement within the function
    guided_filter_params['ridge_filter']['do_ridge_enhance'] = False
    guided_filter_params['ridge_filter']['mix_ratio'] = 0.5 # combine 25% of the ridge enhanced image with the input guide image. 
    guided_filter_params['ridge_filter']['sigmas'] = [2.] # should be approx thickness of protrusions
    
    
    # we use adaptive radius based on the bounding box size of individual cells. 
    guided_filter_params['guide_filter']['min_protrusion_size'] = 15 # this is the minimum radius irregardless of size. 
    guided_filter_params['guide_filter']['adaptive_radius_bool'] = True
    guided_filter_params['guide_filter']['size_factor'] = 1/2. # this is the fraction of the bounding box mean length. about the max before theres interference
    guided_filter_params['guide_filter']['eps'] = 1e-4 # we can increase this to reduce the guided filter strength or decrease to increase.. 
    
    # we use additive mode to infill the hollow regions. 
    guided_filter_params['guide_filter']['mode'] = 'additive' # if the guide image inside is hollow (i.e. black pixels of low intensity like here), you may need to run 'additive' mode and tune 'base_erode' when using large radius
    guided_filter_params['guide_filter']['base_dilate'] = 1
    guided_filter_params['guide_filter']['base_erode'] = 0
    guided_filter_params['guide_filter']['collision_erode'] = 2
    guided_filter_params['guide_filter']['collision_close'] = 3
    guided_filter_params['guide_filter']['collision_dilate'] = 0
    guided_filter_params['guide_filter']['collision_fill_holes'] = True
    
    
    guided_filter_params['guide_filter']['use_int'] = 1 # not sure why but this makes guide filter focus on edges, which is better for this application. 
    
    
    """
    Constructing a customized guide image. 
    """
    # set the input as the preprocessed, this will be augmented by the ridge filtered version internally. 
    guide_image = img_preprocess[...,0].copy()
    
    import skimage.exposure as skexposure
    guide_image = skexposure.equalize_adapthist(guide_image, clip_limit=0.05) # increase the contrast.  #### this seems needed! 
    guide_image = uSegment3D_filters.normalize(guide_image, clip=True)
    
    segmentation3D_filt_guide, guide_image_used = uSegment3D.guided_filter_3D_cell_segmentation_MP(segmentation3D_filt_diffuse,
                                                                                                guide_image=guide_image,
                                                                                                params=guided_filter_params)
        
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Mid Slices Segmentation Overlay (Post guided filter)')
    plt.imshow(sksegmentation.mark_boundaries(np.dstack([guide_image[guide_image.shape[0]//2], 
                                                          guide_image[guide_image.shape[0]//2], 
                                                          guide_image[guide_image.shape[0]//2]]),
                                              segmentation3D_filt_guide[guide_image.shape[0]//2], 
                                              color=(0,1,0), mode='thick'))
    plt.subplot(132)
    plt.imshow(sksegmentation.mark_boundaries(np.dstack([guide_image[:,guide_image.shape[1]//2], 
                                                          guide_image[:,guide_image.shape[1]//2], 
                                                          guide_image[:,guide_image.shape[1]//2]]),
                                              segmentation3D_filt_guide[:,guide_image.shape[1]//2], 
                                              color=(0,1,0), mode='thick'))
    plt.subplot(133)
    plt.imshow(sksegmentation.mark_boundaries(np.dstack([guide_image[:,:,guide_image.shape[2]//2], 
                                                          guide_image[:,:,guide_image.shape[2]//2], 
                                                          guide_image[:,:,guide_image.shape[2]//2]]),
                                              segmentation3D_filt_guide[:,:,guide_image.shape[2]//2], 
                                              color=(0,1,0), mode='thick'))
    plt.savefig(os.path.join(savecellfolder,
                             'segmentation_guide-filter_overlay_image_midslices-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')
    
    
    # save this final segmentation
    uSegment3D_fio.save_segmentation(os.path.join(savecellfolder,
                                                  'uSegment3D_Tcells_labels_postprocess-diffuse-guided_filter.tif'), segmentation3D_filt_guide)
    

    
