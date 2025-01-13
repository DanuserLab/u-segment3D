#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:55:13 2024

@author: s205272
"""

def spectral_unmix_RGB(img, n_components=3, alpha=1., l1_ratio=0.5):
    
    from sklearn.decomposition import NMF
    from skimage.exposure import rescale_intensity
    
    img_vector = img.reshape(-1,img.shape[-1]) / 255.
    color_model = NMF(n_components=n_components, init='nndsvda', random_state=0, l1_ratio=l1_ratio) # Note ! we need a high alpha ->. 
    W = color_model.fit_transform(img_vector)
    
    img_vector_NMF_rgb = W.reshape((img.shape[0], img.shape[1], -1))
    img_vector_NMF_rgb = np.uint8(255*rescale_intensity(img_vector_NMF_rgb))
    
    return img_vector_NMF_rgb, color_model
    

def apply_unmix_model(img, model):
    import numpy as np 
    from skimage.exposure import rescale_intensity
    
    img_vector = img.reshape(-1,img.shape[-1]) / 255.
    img_proj_vector = model.transform(img_vector)
    
    img_proj_vector = img_proj_vector.reshape((img.shape[0], img.shape[1], -1))
    
    return img_proj_vector

def demix_videos(vid1, vid2, l1_ratio=0.5):
    
    import numpy as np 
    
    # vid = np.dstack([im[ref_slice], im_cancer[ref_slice]])
    vid = np.dstack([np.max(vid1, axis=0), 
                     np.max(vid2, axis=0)])
    # vid = np.concatenate([im[...,None], im_cancer[...,None]], axis=-1)
    unmix_img, unmix_model = spectral_unmix_RGB(vid, n_components=2, alpha=1., l1_ratio=l1_ratio)
    mix_components = unmix_model.components_.copy()
    
    
    mix_components_origin = np.argmax(mix_components, axis =1 )
    mix_components_origin_mag = np.max(mix_components, axis =1)
    
    mix_components_origin = mix_components_origin[mix_components_origin_mag>0]
    
    
    NMF_channel_order = []
    NMF_select_channels = []
    select_channels = [0,1]
    for ch in select_channels:
        if ch in mix_components_origin:
            # find the order. 
            NMF_select_channels.append(ch)
            order = np.arange(len(mix_components_origin))[mix_components_origin==ch]
            NMF_channel_order.append(order)
    
    NMF_channel_order = np.hstack(NMF_channel_order)
    NMF_select_channels = np.hstack(NMF_select_channels)
    
    vid = np.concatenate([vid1[...,None], 
                          vid2[...,None]], axis=-1)
    unmixed_vid = np.array([apply_unmix_model(frame, unmix_model) for frame in vid])
    
    # write this to a proper video. 
    unmixed_vid_out = np.zeros_like(vid)
    unmixed_vid_out[...,NMF_select_channels] = unmixed_vid[...,NMF_channel_order]
    
    return unmixed_vid_out
    

if __name__=="__main__":
    
    """
    
    This script will demo how to use u-Segment3D's auto-tuning with pretrained Cellpose 2D models 
    to generate 3D segmentation for the example of zebrafish macrophages. This example is more involved because we need to do nonstandard preprocessing to improve contrast and edge information to be suitable for Cellpose application.
    
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
    
    """
    There are two channels, one cancer, one macrophage. There is spectral overlap. and signal needs to be enhanced for segmentation. 
    """
    cancer_imgfile = '../../example_data/multi_cells/zebrafish_macrophages/1_CH552_000000.tif'
    macrophage_imgfile = '../../example_data/multi_cells/zebrafish_macrophages/1_CH488_000000.tif'
    
    cancer_img = skio.imread(cancer_imgfile)
    macrophage_img = skio.imread(macrophage_imgfile)
    

    # setup a save folder for this example.
    # 1. create a save folder 
    savecellfolder = os.path.join('.', 
                                  'macrophage_segment');
    uSegment3D_fio.mkdir(savecellfolder)
    
    
    """
    Visualize the image in max projection and mid-slice to get an idea of how it looks
    """
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Cancer Max. Projection')
    plt.imshow(cancer_img.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(cancer_img.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(cancer_img.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(savecellfolder,
                             'input_cancer_image_max-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Macrophage Max. Projection')
    plt.imshow(macrophage_img.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(macrophage_img.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(macrophage_img.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(savecellfolder,
                             'input_macrophage_image_max-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    

    
    # =============================================================================
    #     0. Custom Preprocess of the image channels. 
    # =============================================================================
    
    zoom = [1., 1/3.42, 1/3.42]
    
    # the samples are not isotropic. therefore we need to resize. Since xy is already large number of pixels, we downsize this instead of padding z (first axis)
    cancer_img = ndimage.zoom(cancer_img, zoom, order=1, mode='reflect')
    macrophage_img = ndimage.zoom(macrophage_img, zoom, order=1, mode='reflect')
    
    cancer_img = uSegment3D_filters.normalize(cancer_img, pmin=2, pmax=99.8, clip=True)
    macrophage_img = uSegment3D_filters.normalize(macrophage_img, pmin=2, pmax=99.8, clip=True)

    """
    do background correction subtractively. 
    """
    bg_ds = 5 

    # estimate background
    im_cancer_bg = uSegment3D_filters.smooth_vol(cancer_img, ds=bg_ds, smooth=5)
    im_macrophage_bg = uSegment3D_filters.smooth_vol(macrophage_img, ds=bg_ds, smooth=5)
    
    # debackground and renormalize
    cancer_img = uSegment3D_filters.normalize(cancer_img-im_cancer_bg, pmin=2, pmax=99.8, clip=True)
    macrophage_img = uSegment3D_filters.normalize(macrophage_img-im_macrophage_bg, pmin=2, pmax=99.8, clip=True)

    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Cancer Max. Projection after background subtract and resize')
    plt.imshow(cancer_img.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(cancer_img.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(cancer_img.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(savecellfolder,
                             'bg-subtract_and_resize_input_cancer_image_max-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Macrophage Max. Projection after background subtract and resize')
    plt.imshow(macrophage_img.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(macrophage_img.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(macrophage_img.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(savecellfolder,
                             'bg-subtract_and_resize_input_macrophage_image_max-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    

    """
    Read the synthetic PSF to perform deconvolution enhancement. This PSF can be generally used with lightsheet microscopes
    """
    import scipy.io as spio 
    
    psf_meSPIM = '../../PSFs/meSPIM_PSF_kernel.mat'
    psf_meSPIM = spio.loadmat(psf_meSPIM)['PSF']
    psf_meSPIM = psf_meSPIM / (float(np.sum(psf_meSPIM))) # normalize the PSF. 
    
    import skimage.restoration as skrestoration 
    
    im_cancer_deconv = skrestoration.wiener(cancer_img, psf_meSPIM, balance=0.1) # was doing balance=0.5 # use a smaller balance to retain sharper features. 
    im_cancer_deconv = np.clip(im_cancer_deconv, 0, 1) 
    im_cancer_deconv = np.array([ uSegment3D_filters.anisodiff(ss, niter=15,kappa=1,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False) for ss in im_cancer_deconv]) 
    
    
    im_macrophage_deconv = skrestoration.wiener(macrophage_img, psf_meSPIM, balance=0.1) # was doing balance=0.5 # use a smaller balance to retain sharper features. 
    im_macrophage_deconv = np.clip(im_macrophage_deconv, 0, 1) 
    im_macrophage_deconv = np.array([ uSegment3D_filters.anisodiff(ss, niter=15,kappa=1,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False) for ss in im_macrophage_deconv]) 
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Cancer Max. Projection after deconv, background subtract and resize')
    plt.imshow(im_cancer_deconv.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(im_cancer_deconv.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(im_cancer_deconv.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(savecellfolder,
                             'deconv_bg-subtract_and_resize_input_cancer_image_max-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Macrophage Max. Projection after deconv, background subtract and resize')
    plt.imshow(im_macrophage_deconv.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(im_macrophage_deconv.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(im_macrophage_deconv.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(savecellfolder,
                             'deconv_bg-subtract_and_resize_input_macrophage_image_max-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    
    """
    Perform a non-negative matrix factorization spectral unmixing. 
    """
    im_unmix = demix_videos(im_cancer_deconv, 
                            im_macrophage_deconv, 
                            l1_ratio=0.5)
    
    # rescale. 
    im_cancer_deconv = uSegment3D_filters.normalize(im_unmix[...,0], clip=True)
    im_macrophage_deconv = uSegment3D_filters.normalize(im_unmix[...,1], clip=True)
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Cancer Max. Projection after unmix, deconv, background subtract and resize')
    plt.imshow(im_cancer_deconv.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(im_cancer_deconv.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(im_cancer_deconv.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(savecellfolder,
                             'unmix_deconv_bg-subtract_and_resize_input_cancer_image_max-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Macrophage Max. Projection after unmix, deconv, background subtract and resize')
    plt.imshow(im_macrophage_deconv.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(im_macrophage_deconv.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(im_macrophage_deconv.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(savecellfolder,
                             'unmix_deconv_bg-subtract_and_resize_input_macrophage_image_max-projection.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    
    img_preprocess = im_macrophage_deconv.copy()
   
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
    # cellpose_segment_params['cellpose_modelname'] = 'cyto3' # seems to be best model? 
    
    """
    If the below auto-inferred diameter is picking up noise and being too small, we can increase the default ksize, alternatively we can add median_filter.
    """
    cellpose_segment_params['ksize'] = 21
    
    # invert model
    cellpose_segment_params['model_invert'] = False
    cellpose_segment_params['use_edge'] = True # for autofinding best slice. 
    
    
    # this expects a multichannel input image and in the format (M,N,L,channels) i.e. channels last.
    if len(img_preprocess.shape) == 3:
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
    aggregation_params['gradient_descent']['gradient_decay'] = 0.01 # increase this to alleviate splitting of elongated/branching structures with cellpose gradients or decrease towards 0.0 to encourage splitting. 
    aggregation_params['gradient_descent']['momenta'] = 0.98 # help boost the splitting 
    
    
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
                                                  'uSegment3D_macrophage_labels.tif'), segmentation3D)
    
    
    # 3. if you want to save the intermediate combined probability and gradients, we recommend using pickle 
    uSegment3D_fio.write_pickle(os.path.join(savecellfolder,
                                                  'uSegment3D_macrophage_3Dcombined_probs_and_gradients'), 
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
                                                  'uSegment3D_macrophages_labels_postprocess.tif'), segmentation3D_filt)
    
    
    
    """
    2. second postprocessing is to enhance the segmentation. 
        a) diffusing the labels to improve concordance with image boundaries 
        b) guided filter to recover detail such as subcellular protrusions on the surface of the cell. 
    """
    
    ###### a) label diffusion.
    label_diffusion_params = uSegment3D_params.get_label_diffusion_params()
    label_diffusion_params['diffusion']['refine_iters'] = 25 
    label_diffusion_params['diffusion']['refine_alpha'] = 0.25 
    label_diffusion_params['diffusion']['refine_clamp'] = 0.75
    
    
    # customize the diffusion image. 
    guide_image = img_preprocess[...,0].copy()
    
    import skimage.exposure as skexposure
    
    guide_image = ndimage.gaussian_filter(guide_image, sigma=1) 
    guide_image = skexposure.adjust_gamma(guide_image, gamma=0.4) # increase the contrast.  
    guide_image = uSegment3D_filters.normalize(guide_image, clip=True)
    guide_image = uSegment3D_filters.guidedfilter(guide_image, 
                                                  guide_image, radius=5, eps=0.01)
    
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
                                                  'uSegment3D_macrophages_labels_postprocess-diffuse.tif'), segmentation3D_filt_diffuse)
    
    
    
    ##### b) guided filter, (but we are going to do this at original resolution)
    guided_filter_params = uSegment3D_params.get_guided_filter_params()
    
    #  for guided image we will use the input to Cellpose and specify we want to further do ridge filter enhancement within the function
    guided_filter_params['ridge_filter']['do_ridge_enhance'] = False
    guided_filter_params['ridge_filter']['mix_ratio'] = 0.5 # combine 25% of the ridge enhanced image with the input guide image. 
    guided_filter_params['ridge_filter']['sigmas'] = [2.] # should be approx thickness of protrusions
    
    
    # we use adaptive radius based on the bounding box size of individual cells. 
    guided_filter_params['guide_filter']['min_protrusion_size'] = 25 # this is the minimum radius irregardless of size. 
    guided_filter_params['guide_filter']['adaptive_radius_bool'] = True
    guided_filter_params['guide_filter']['size_factor'] = 2/3. # this is the fraction of the bounding box mean length. about the max before theres interference
    guided_filter_params['guide_filter']['eps'] = 1e-4 # we can increase this to reduce the guided filter strength or decrease to increase.. 
    
    # we use additive mode to infill the hollow regions. 
    guided_filter_params['guide_filter']['mode'] = 'normal' # if the guide image inside is hollow (i.e. black pixels of low intensity like here), you may need to run 'additive' mode and tune 'base_erode' when using large radius
    guided_filter_params['guide_filter']['base_dilate'] = 1
    guided_filter_params['guide_filter']['base_erode'] = 0
    guided_filter_params['guide_filter']['collision_erode'] = 1
    guided_filter_params['guide_filter']['collision_close'] = 2
    guided_filter_params['guide_filter']['collision_dilate'] = 0
    guided_filter_params['guide_filter']['collision_fill_holes'] = True
    
    
    guided_filter_params['guide_filter']['use_int'] = False # not sure why but this makes guide filter focus on edges, which is better for this application. 
    
    
    """
    Constructing a customized guide image. 
    """
    # set the input as the preprocessed, this will be augmented by the ridge filtered version internally. 
    guide_image = img_preprocess[...,0].copy()
    
    import skimage.exposure as skexposure
    # guide_image = skexposure.equalize_adapthist(guide_image, clip_limit=0.05) # increase the contrast.  #### this seems needed! 
    # guide_image = uSegment3D_filters.normalize(guide_image, clip=True)
    guide_image = uSegment3D_filters.guidedfilter(guide_image, 
                                                  guide_image, radius=5, eps=0.01) # smooth the guide. 
    
    
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
                                                  'uSegment3D_macrophages_labels_postprocess-diffuse-guided_filter.tif'), segmentation3D_filt_guide)
    
    
    
    
