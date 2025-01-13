#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:55:13 2024

@author: s205272
"""

if __name__=="__main__":
    
    """
    
    This script will demo how to use u-Segment3D's auto-tuning with pretrained Cellpose 2D models 
    to generate 3D segmentation for a single cell with surface ruffles.
    
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
    imgfile = '../../example_data/single_cells/ruffles/rigid_1_CH00_000000.tif'
    
    img = skio.imread(imgfile)
    
    
    """
    Define save folder for outputs
    """
    savecellfolder = os.path.join('.', 
                                  'ruffles_segment');
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
    #     1. Preprocess the image. 
    # =============================================================================
    
    # generate the default preprocessing parameter settings. 
    preprocess_params = uSegment3D_params.get_preprocess_params()
    
    # this image is isotropic, therefore we don't change any of the parameters
    print('========== preprocessing parameters ========')
    print(preprocess_params)    
    print('============================================')
    
    """
    For this example which is deconvolved we found background illumination correction made it worse.
    """
    preprocess_params['do_bg_correction'] = True
    
    """
    we will run segmentation at the same scale as the image. 
    """
    preprocess_params['factor'] = 0.5 #wasn't good. 
    # preprocess_params['normalize_min'] = 5.0 # increase the lowest value
    
    """
    The cell is cropped a little close. We are going to pad 
    """
    pad_size=25
    img = np.pad(img, [[pad_size,pad_size], [pad_size, pad_size], [pad_size, pad_size]], mode='constant')
    img[img==0] = np.nanmedian(img[img>0])
    
    img = ndimage.median_filter(img, size=5)
    
    
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
    
    # cellpose_segment_params['cellpose_modelname'] = 'cyto' # try different models like 'cyto', 'cyto3' for cells, and additionally 'nuclei' for nuclei
    cellpose_segment_params['cellpose_modelname'] = 'cyto2' # try different models like 'cyto', 'cyto3' for cells, and additionally 'nuclei' for nuclei
    # cellpose_segment_params['cellpose_modelname'] = 'cyto3'
    
    """
    If the below auto-inferred diameter is picking up noise and being too small, we can increase the default ksize, alternatively we can add median_filter.
    """
    cellpose_segment_params['ksize'] = 25
    
    
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
    
    # add some temporal decay 
    aggregation_params['gradient_descent']['gradient_decay'] = 0.1 # increase this to alleviate splitting of elongated/branching structures with cellpose gradients or decrease towards 0.0 to encourage splitting. 
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
                                                  'uSegment3D_ruffle_labels.tif'), segmentation3D)
    
    
    # 3. if you want to save the intermediate combined probability and gradients, we recommend using pickle 
    uSegment3D_fio.write_pickle(os.path.join(savecellfolder,
                                                  'uSegment3D_ruffle_3Dcombined_probs_and_gradients'), 
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
    
    # since this is single cell we combine any fragments and take the largest component, whereby previous steps were used to find cells.
    segmentation3D_filt = (segmentation3D_filt>0)
    segmentation3D_filt = uSegment3D_filters.largest_component_vol(segmentation3D_filt)*1
    
    
    ###### a) label diffusion.
    label_diffusion_params = uSegment3D_params.get_label_diffusion_params()
    label_diffusion_params['diffusion']['refine_iters'] = 25 # use lower amount of iterations, since base segmentation is quite close. 
    label_diffusion_params['diffusion']['refine_alpha'] = 0.5 
    label_diffusion_params['diffusion']['refine_clamp'] = 0.75
    
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
                                                  'uSegment3D_ruffle_labels_postprocess-diffuse.tif'), segmentation3D_filt_diffuse)
    
    
    
    ##### b) guided filter, (but we are going to do this at original resolution)
    guided_filter_params = uSegment3D_params.get_guided_filter_params()
    
    #  for guided image we will use the input to Cellpose and specify we want to further do ridge filter enhancement within the function
    guided_filter_params['ridge_filter']['do_ridge_enhance'] = True
    guided_filter_params['ridge_filter']['mix_ratio'] = 0.5 # combine 25% of the ridge enhanced image with the input guide image. 
    guided_filter_params['ridge_filter']['sigmas'] = [1.8] # thicker.
    
    # we can adjust the radius to try to capture more detail 
    guided_filter_params['guide_filter']['radius'] = 65 # drop lower (this should be approx size of protrusions) 
    guided_filter_params['guide_filter']['eps'] = 1e-4 # we can increase this to reduce the guided filter strength or decrease to increase.. 
    guided_filter_params['guide_filter']['mode'] = 'additive' # if the guide image inside is hollow (i.e. black pixels of low intensity like here), you may need to run 'additive' mode and tune 'base_erode' when using large radius
    guided_filter_params['guide_filter']['base_erode'] = 2
    guided_filter_params['guide_filter']['collision_erode'] = 0
    guided_filter_params['guide_filter']['collision_close'] = 0
    guided_filter_params['guide_filter']['collision_fill_holes'] = True
    
    
    try:
        # for image we use linear interpolation i.e. order=1
        guide_image = uSegment3D_gpu.dask_cuda_rescale(img_preprocess[...,0],
                                                       zoom=[1./preprocess_params['factor'],1./preprocess_params['factor'],1./preprocess_params['factor']],
                                                       order=1,
                                                       mode='reflect',
                                                       chunksize=(512,512,512)) # note the inversion of the original factor
    except:
        guide_image = uSegment3D_gpu.zoom_3d_pytorch(img_preprocess[...,0],
                                                       zoom_factors = [1./preprocess_params['factor'],1./preprocess_params['factor'],1./preprocess_params['factor']]) # note the inversion of the original factor
        
        
    # guide_image = ndimage.median_filter(guide_image, 
    #                                     size=3) # use this filtered version 
    
    try:
        # for segmentation we use 0th order i.e. order=0 nearest-neighbor interpolation to maintain integer-values
        segmentation3D_filt_diffuse = uSegment3D_gpu.dask_cuda_rescale(segmentation3D_filt_diffuse,
                                                                       zoom=[1./preprocess_params['factor'],1./preprocess_params['factor'],1./preprocess_params['factor']],
                                                                       order=0,
                                                                       mode='reflect',
                                                                       chunksize=(512,512,512)).astype(np.int32) # note the inversion of the original factor
    except:
        # for segmentation we use 0th order i.e. order=0 nearest-neighbor interpolation to maintain integer-values
        segmentation3D_filt_diffuse = uSegment3D_gpu.zoom_3d_pytorch(segmentation3D_filt_diffuse,
                                                                     zoom_factors = [1./preprocess_params['factor'],1./preprocess_params['factor'],1./preprocess_params['factor']],
                                                                     interp_mode='nearest').astype(np.int32) # note the inversion of the original factor


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
                                                  'uSegment3D_ruffle_labels_postprocess-diffuse-guided_filter.tif'), segmentation3D_filt_guide)
    

    """
    We assume all segmented regions are part of a single cell and we will extract its surface mesh and measure mesh properties. 
    For more extensive mesh processing you should check out u-Unwrap3D (https://github.com/DanuserLab/u-unwrap3D/blob/master/unwrap3D)
    
    The mesh is saved as a .obj file which can be opened by mesh programs such as meshlab, chimeraX 
    """
    import segment3D.meshtools as uSegment3D_meshtools
    
    cell_binary = segmentation3D_filt_guide>0
    cell_binary = uSegment3D_filters.largest_component_vol(cell_binary, connectivity=2)
    
    cell_surface_mesh = uSegment3D_meshtools.marching_cubes_mesh_binary(cell_binary.transpose(2,1,0),
                                                                        presmooth=1.,
                                                                        contourlevel=0.5,
                                                                        keep_largest_only=True) # the transpose is to make the mesh visualize same orientation as that via ImageJ Volume viewer. 
    
    # check orientation of mesh, if opposite winding, reverse the order of the faces.
    if np.sign(cell_surface_mesh.volume) < 0: 
        cell_surface_mesh.faces = cell_surface_mesh.faces[:,::-1]
    
    cell_surface_mesh_props = uSegment3D_meshtools.measure_props_trimesh(cell_surface_mesh, 
                                                                         main_component=True,
                                                                         clean=True)
    
    print('=============')
    print('Final segmentation surface properties')
    print('=============')
    print(cell_surface_mesh_props)
    
    tmp = cell_surface_mesh.export(os.path.join(savecellfolder,
                                                  'uSegment3D_ruffle_labels_postprocess-diffuse-guided_filter_surface-mesh.obj'))
                                   
                                   
                                   
                                   
    
