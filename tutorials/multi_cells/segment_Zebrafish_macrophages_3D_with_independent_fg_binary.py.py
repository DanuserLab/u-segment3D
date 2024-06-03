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
    
    The difference between this script and the other example is to show how to insert an independent foreground binary into the script but still re-use the cellpose predicted gradients for 2D-to-3D aggregation. 
    
    We will do this by applying Otsu thresholding to compute an independent binary. 
    
    An advantage of this method is its easy to train a 3D foreground/background segmentation which might be more precise. 
    
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
    plt.show()
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Macrophage Max. Projection')
    plt.imshow(macrophage_img.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(macrophage_img.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(macrophage_img.max(axis=2), cmap='gray')
    plt.show()
    
    
    
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
    plt.show()
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Macrophage Max. Projection after background subtract and resize')
    plt.imshow(macrophage_img.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(macrophage_img.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(macrophage_img.max(axis=2), cmap='gray')
    plt.show()
    

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
    plt.show()
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Macrophage Max. Projection after deconv, background subtract and resize')
    plt.imshow(im_macrophage_deconv.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(im_macrophage_deconv.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(im_macrophage_deconv.max(axis=2), cmap='gray')
    plt.show()
    
    
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
    plt.show()
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Macrophage Max. Projection after unmix, deconv, background subtract and resize')
    plt.imshow(im_macrophage_deconv.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(im_macrophage_deconv.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(im_macrophage_deconv.max(axis=2), cmap='gray')
    plt.show()
    
    
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
    img_preprocess = img_preprocess[...,None] # we generate an artificial channel
    
    
    #### 1. running for xy orientation. If the savefolder and basename are specified, the output will be saved as .pkl and .mat files 
    img_segment_2D_xy_diams, img_segment_3D_xy_probs, img_segment_2D_xy_flows, img_segment_2D_xy_styles = uSegment3D.Cellpose2D_model_auto(img_preprocess, 
                                                                                                                                            view='xy', 
                                                                                                                                            params=cellpose_segment_params, 
                                                                                                                                            basename=None, savefolder=None)

    #### 2. running for xz orientation 
    img_segment_2D_xz_diams, img_segment_3D_xz_probs, img_segment_2D_xz_flows, img_segment_2D_xz_styles = uSegment3D.Cellpose2D_model_auto(img_preprocess, 
                                                                                                                                            view='xz', 
                                                                                                                                            params=cellpose_segment_params, 
                                                                                                                                            basename=None, savefolder=None)

    #### 3. running for yz orientation 
    img_segment_2D_yz_diams, img_segment_3D_yz_probs, img_segment_2D_yz_flows, img_segment_2D_yz_styles = uSegment3D.Cellpose2D_model_auto(img_preprocess, 
                                                                                                                                            view='yz', 
                                                                                                                                            params=cellpose_segment_params, 
                                                                                                                                            basename=None, savefolder=None)

    
    # =============================================================================
    #     3. Compute a foreground binary (using adapted Otsu thresholding) on the full 3D image 
    # =============================================================================
    import skimage.filters as skfilters
    import skimage.morphology as skmorph
    
    im_combine = img_preprocess[...,0].copy()
    im_edges = im_combine - ndimage.gaussian_filter(im_combine, sigma=1) # enhance edges
    im_ = (im_combine-im_combine.mean()) / (np.std(im_combine)) ; im_ = np.clip(im_, 0, 4)
    binary_thresh = skfilters.threshold_otsu(im_) # what the best threshold?  # this is bad 

    im_binary = im_ > binary_thresh #+0.1;
    im_binary = skmorph.remove_small_objects(im_binary, min_size=1000, connectivity=2)
    
    # hm seems we still need to grab the higher frequency signals? 
    comb = np.maximum(im_binary*1, (im_edges-im_edges.mean())/(4*np.std(im_edges)))
    
    
    im_binary = comb >= 1
    im_binary = skmorph.binary_closing(im_binary, skmorph.ball(1))
    im_binary = ndimage.binary_fill_holes(im_binary)
    im_binary = skmorph.binary_erosion(im_binary, skmorph.ball(1))  
    im_binary = skmorph.remove_small_objects(im_binary, min_size=250, connectivity=2)
    
    
    # save this binary 
    # we can save the segmentation with its colors using provided file I/O functions in u-Segment3D. If the savefolder exists and provided with basename in the function above, these would be saved automatically. 
    # 1. create a save folder 
    savecellfolder = os.path.join('.', 
                                  'macrophage_segment_with_independent_fg_binary');
    uSegment3D_fio.mkdir(savecellfolder)
    
    # 2. joint the save folder with the filename we wish to use. 2 files will be saved, 1=segmentation as labels and 2nd = 16color RGB colored segmentation for visualization
    skio.imsave(os.path.join(savecellfolder,
                             'Binary_segment_3D_macrophage_foreground.tif'), np.uint8(255*im_binary))
    
    
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(231)
    plt.title('Input Macrophage Max. Projection')
    plt.imshow(im_macrophage_deconv.max(axis=0), cmap='gray')
    plt.subplot(232)
    plt.imshow(im_macrophage_deconv.max(axis=1), cmap='gray')
    plt.subplot(233)
    plt.imshow(im_macrophage_deconv.max(axis=2), cmap='gray')
    
    plt.subplot(234)
    plt.title('Binary segment Macrophage Max. Projection')
    plt.imshow(im_binary.max(axis=0), cmap='gray')
    plt.subplot(235)
    plt.imshow(im_binary.max(axis=1), cmap='gray')
    plt.subplot(236)
    plt.imshow(im_binary.max(axis=2), cmap='gray')
    plt.show()
    
    
    # =============================================================================
    #     4. We can now use the predicted probabilies and flows directly to aggregate a 3D consensus segmentation (Direct Method), passing the precomputed foreground binary
    # =============================================================================
    
    aggregation_params = uSegment3D_params.get_2D_to_3D_aggregation_params()
    
    print('========== Default 2D-to-3D aggregation parameters ========')
    print(aggregation_params)    
    print('============================================')
    
    # make sure that we say we are using Cellpose probability predictions which needs to be normalized. If not using Cellpose predicted masks, then set this to be False. We assume this has been appropriately normalized to 0-1
    aggregation_params['combine_cell_probs']['cellpose_prob_mask'] = True 
    
    # add some temporal decay 
    aggregation_params['gradient_descent']['gradient_decay'] = 0.1 # increase this to alleviate splitting of elongated/branching structures with cellpose gradients or decrease towards 0.0 to encourage splitting. 
    aggregation_params['gradient_descent']['momenta'] = 0.98 # help boost the splitting 
    
    
    # probs and gradients should be supplied in the order of [xy, xz, yz]. if one or more do not exist use [] e.g. [xy, [], []]
    segmentation3D, (probability3D, gradients3D) = uSegment3D.aggregate_2D_to_3D_segmentation_direct_method(probs=[[],
                                                                                                                    [],
                                                                                                                    []], 
                                                                                                            gradients=[img_segment_2D_xy_flows,
                                                                                                                        img_segment_2D_xz_flows,
                                                                                                                        img_segment_2D_yz_flows], 
                                                                                                            params=aggregation_params,
                                                                                                            precombined_binary = im_binary, 
                                                                                                            precombined_gradients = None,
                                                                                                            savefolder=None,
                                                                                                            basename=None)
    
    # we can save the segmentation with its colors using provided file I/O functions in u-Segment3D. If the savefolder exists and provided with basename in the function above, these would be saved automatically. 
    # 1. create a save folder 
    savecellfolder = os.path.join('.', 
                                  'macrophage_segment_with_independent_fg_binary');
    uSegment3D_fio.mkdir(savecellfolder)
    
    # 2. joint the save folder with the filename we wish to use. 2 files will be saved, 1=segmentation as labels and 2nd = 16color RGB colored segmentation for visualization
    uSegment3D_fio.save_segmentation(os.path.join(savecellfolder,
                                                  'uSegment3D_macrophage_labels_with_independent-foreground.tif'), segmentation3D)
    
    
    # 3. if you want to save the intermediate combined probability and gradients, we recommend using pickle 
    uSegment3D_fio.write_pickle(os.path.join(savecellfolder,
                                                  'uSegment3D_macrophage_3Dcombined_probs_and_gradients_with_independent-foreground'), 
                                savedict={'probs': probability3D.astype(np.float32),
                                          'gradients': gradients3D.astype(np.float32)})
    
    
    