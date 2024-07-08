# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:37:54 2023

@author: fyz11

helper functions for applying several standard DL segmentation models. 
"""

#import cupy 
import numpy as np 



def reorder_labels_2D(labels, bg=0):
    
    import skimage.measure as skmeasure
    
    regprops = skmeasure.regionprops(labels)
    
    reg_labels = np.setdiff1d(np.unique(labels),bg)
    centroids = np.vstack([re.centroid for re in regprops])
    
    ind = np.lexsort((centroids[:,1], centroids[:,0]))
    labels_out = np.zeros_like(labels)
    
    for ii, lab in enumerate(ind):
        labels_out[labels==reg_labels[lab]] = ii + 1 # since this is not background.
        
    return labels_out


def reorder_labels_3D(labels, bg=0):
    
    import skimage.measure as skmeasure
    
    regprops = skmeasure.regionprops(labels)
    
    reg_labels = np.setdiff1d(np.unique(labels),bg)
    centroids = np.vstack([re.centroid for re in regprops])
    
    ind = np.lexsort((centroids[:,2], centroids[:,1], centroids[:,0]))
    labels_out = np.zeros_like(labels)
    
    for ii, lab in enumerate(ind):
        labels_out[labels==reg_labels[lab]] = ii + 1 # since this is not background.
        
    return labels_out


def _BackgroundRemoval(im):
    # given an image, utilises the pixels to subtract the background in the image 
    # remembering images are represented by a matrix with x axis = cols, y axis = rows,
    # The algorithm carries out the following steps:
    # 1) normalize the input image
    # 2) fits a 2nd order background model 
    # 3) subtracts this from the input image, then makes everything positive to be a proper image, 
    # 4) renormalises the image, 

    import numpy as np

    nrows,ncols = im.shape    
    N = nrows*ncols
    
    xx, yy = np.meshgrid(np.arange(0,ncols),np.arange(0,nrows))
    xx = xx.ravel()
    yy = yy.ravel()
    
    X =np.column_stack( (np.ones(N), xx, yy, np.square(xx), np.multiply(xx,yy), np.square(yy)) ) 
    p = np.linalg.lstsq(X,im.ravel()) #X\im(:)
    
    p = p[0] 

    temp = im.ravel() - np.dot(X,p)
    imgInit = temp.reshape(nrows,ncols)

    # visualize background. 
    # plt.figure()
    # plt.imshow( np.reshape(temp, im.shape), cmap='gray')
    # plt.show(block=False)

    if np.min(imgInit) == np.max(imgInit):
        imgInit = np.zeros(im.shape)
    else:
        imgInit = (imgInit - np.min(imgInit))/(np.max(imgInit) - np.min(imgInit))
  
    # imgInit is the subtracted background (flattened image), normalised to between [0,1]
    return imgInit



def _ma_average(sig, winsize=3, mode='reflect', avg_func=np.nanmean):
    
    sig_ = np.pad(sig, [winsize//2, winsize//2], mode=mode)
    sig_out = []
    for ii in np.arange(len(sig)):
        data = sig_[ii:ii+winsize]
        sig_out.append(avg_func(data))
    return np.hstack(sig_out)
    
def _histeq_low_contrast(im, kernel_size=(256,256), clip_limit=0.01, fraction_threshold=0.1):
    
    import skimage.exposure as skexposure
    
    if skexposure.is_low_contrast(im, fraction_threshold):
        return im
    else:
        # print('equalizing image')
        try:
            from .gpu import cuda_equalize_adapthist
            return cuda_equalize_adapthist(im, kernel_size=kernel_size, clip_limit=clip_limit)
        except:
            return skexposure.equalize_adapthist(im, kernel_size=kernel_size, clip_limit=clip_limit)

# def apply_cellpose_model_2D(im_stack, model, 
#                             model_channels, 
#                             best_diam=None, 
#                             model_invert=False, 
#                             test_slice=None, 
#                             diam_range=np.arange(15,51,5), 
#                             ksize=25,
#                             smoothwinsize=5,
#                             hist_norm=True, 
#                             kernel_size=(256,256),
#                             clip_limit=0.01,
#                             fraction_threshold=0.1,
#                             n_proc=48):
    
#     from .filters import var_filter
#     from tqdm import tqdm 
#     import pylab as plt 
#     from skimage.filters.rank import entropy
#     import skimage.morphology as skmorph
#     import skimage.exposure as skexposure
    
#     if best_diam is None:
#         diam_range = np.hstack(diam_range)
        
#         if test_slice is None:
#             # don't auto into the mid slice ---> instead pick the slice with most signal. 
#             if len(im_stack.shape)>3:
#                 signal_stack = [np.nanmedian(np.max(im_slice,axis=-1)) for im_slice in im_stack]
#             else:
#                 print('single channel')
#                 signal_stack = [np.nanmedian(im_slice) for im_slice in im_stack]
#             # test_slice = len(im_stack) // 2
#             test_slice = np.argmax(signal_stack)
#             print(test_slice, len(im_stack))
        
#         diam_score = []
        
#         for diam in diam_range[:]: 
            
#             img = im_stack[test_slice].copy()
#             if hist_norm:
#                 # this messes up ? 
#                 img = _histeq_low_contrast(img, kernel_size=kernel_size, clip_limit=clip_limit, fraction_threshold=fraction_threshold)
#                 # img = normalize(img, pmin=2, pmax=99.8, clip=True)
#                 img = skexposure.rescale_intensity(img)
#             mask, flow, style, diam = model.eval([img], 
#                                                 channels=model_channels, 
#                                                 batch_size=32,
#                                                 do_3D=False, 
#                                                 flow_threshold=0.6,
#                                                 diameter=diam, # this is ok 
#                                                 invert=model_invert) # try inverting?  
        
#             # score the content! e.g. sobel, var, 
#             # prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize))
#             prob = flow[0][2]
#             prob = 1./(1+np.exp(-prob))
#             # prob_score = np.nanmean(1./(var_filter(prob, ksize=ksize) + 0.1)*prob)
#             # prob_score = np.nanmean(var_filter(prob, ksize=ksize))
#             # prob_score = np.median(prob) / np.std(prob)
#             # prob_score = np.nanmean(entropy(prob, skmorph.disk(ksize)))
#             # prob_score = np.mean(prob) / (np.std(prob)) # signal to noise ratio. 
#             # prob_score = np.nanmean(entropy(flow[0][1][0]/5., skmorph.disk(ksize))) + np.nanmean(entropy(flow[0][1][1]/5., skmorph.disk(ksize)))
#             prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize)) # doesn't work weel 
#             # prob_score = np.nanmedian(var_filter(flow[0][1][0], ksize=ksize))+np.nanmedian(var_filter(flow[0][1][1], ksize=ksize))
            
#             # prob_score = mask[0].max()
#             # prob_score = np.nanmean(flow[0][2])
#             diam_score.append(prob_score)
            
#             plt.figure(figsize=(5,5))
#             plt.subplot(311)
#             plt.title(str(diam)+ ' '+str(prob_score))
#             plt.imshow(img[:1024,:1024])
#             plt.subplot(312)
#             plt.title(np.mean(flow[0][2]))
#             plt.imshow(flow[0][2][:1024,:1024])
#             plt.subplot(313)
#             plt.imshow(var_filter(flow[0][2], ksize=ksize)[:1024,:1024])
#             plt.show(block=False)
            
#         diam_score = np.hstack(diam_score)
        
#         # smooth this. 
#         diam_score = _ma_average(diam_score, winsize=smoothwinsize)
            
#         # =============================================================================
#         #     Compute the best. diameter for this view. 
#         # =============================================================================
#         best_diam = diam_range[np.argmax(diam_score)]
        
#         plt.figure()
#         plt.plot(diam_range, diam_score, 'o-')
#         plt.show(block=False)
        
#     print('auto determine cell diameter: ', best_diam)

#     import dask
#     from dask.distributed import Client
    
#     # # n_proc = 48
#     # client = Client(n_workers=n_proc)
#     client = Client()
#     print(client.cluster)
    
#     def run_cp_model(im_, model, channels, batch_size, do_3D, flow_threshold, diameter, invert):
        
#         mask, flow, style, diam = model.eval([im_], 
#                                             channels=model_channels, 
#                                             batch_size=batch_size,
#                                             do_3D=do_3D, 
#                                             flow_threshold=flow_threshold,
#                                             diameter=diameter, # this is ok 
#                                             invert=invert) # try inverting? 
        
#         return np.concatenate([mask[0][None,...], 
#                                flow[0][1], 
#                                flow[0][2][None,...]], axis=0).astype(np.float32)
    
#     # lazy_model = dask.delayed(model.eval)
#     lazy_model = dask.delayed(run_cp_model)
    
    
#     client.scatter(im_stack)
#     future = client.map(lazy_model, im_stack, 
#                                 model=model,
#                                 channels=model_channels, 
#                                 batch_size=32,
#                                 do_3D=False, 
#                                 flow_threshold=0.6,
#                                 diameter=best_diam, # this is ok 
#                                 invert=model_invert)
#     return client.gather(future)
    
#     # # with Client() as client:
#     # # all_masks = []
#     # # all_probs = []
#     # # all_flows = []
#     # # all_styles = []
#     # # all_diams = []
#     # all_results = []
#     # # can we use dask for this? and do cpu .
#     # # for zz in tqdm(np.arange(len(im_stack))):
#     # for zz in np.arange(len(im_stack)):
        
#     #     # mask, flow, style, diam = model.eval([im_stack[zz]], 
#     #     #                                         channels=model_channels, 
#     #     #                                         batch_size=32,
#     #     #                                         do_3D=False, 
#     #     #                                         flow_threshold=0.6,
#     #     #                                         diameter=best_diam, # this is ok 
#     #     #                                         invert=model_invert) # try inverting?  
#     #     # res = lazy_model.eval([im_stack[zz]], 
#     #     #                         channels=model_channels, 
#     #     #                         batch_size=32,
#     #     #                         do_3D=False, 
#     #     #                         flow_threshold=0.6,
#     #     #                         diameter=best_diam, # this is ok 
#     #     #                         invert=model_invert) # try inverting?  
        
#     #     res = lazy_model.eval(im_stack[zz].astype(np.float32), 
#     #                             channels=model_channels, 
#     #                             batch_size=32,
#     #                             do_3D=False, 
#     #                             flow_threshold=0.6,
#     #                             diameter=best_diam, # this is ok 
#     #                             invert=model_invert) # try inverting?  
        
#     #     # all_masks.append(mask[0])
#     #     # all_probs.append(flow[0][2])
#     #     # all_flows.append(flow[0][1])
#     #     # all_styles.append(style[0])
#     #     # all_diams.append(diam)
#     #     all_results.append(res)
        
#     # # all_masks = np.array(all_masks, dtype=np.uint16)
#     # # all_probs = np.array(all_probs, dtype=np.float32)
#     # # all_flows = np.array(all_flows, dtype=np.float32)
#     # # all_styles = np.array(all_styles, dtype=np.float32)
#     # # all_diams = np.array(all_diams, dtype=np.float32)
#     # all_results = dask.compute(all_results)
    
#     print('done')
#     print(len(all_results))
#     #     # all_masks, all_probs, all_flows, all_styles, all_diams = dask.compute(all_masks, all_probs, all_flows, all_styles, all_diams)
#     #     # results = dask.compute(*results)
#     #     # print(len(all_masks))
#     #     # client.close()
#     #     all_masks = np.array(all_masks, dtype=np.uint16)
#     #     all_probs = np.array(all_probs, dtype=np.float32)
#     #     all_flows = np.array(all_flows, dtype=np.float32)
#     #     all_styles = np.array(all_styles, dtype=np.float32)
#     #     all_diams = np.array(all_diams, dtype=np.float32)
    
#     # # client.close()
#     return all_results
#     # return (diam_range, diam_score, best_diam), (all_masks, all_probs, all_flows, all_styles, all_diams)


def apply_cellpose_model_2D_prob_slice(im_slice,
                                       model,
                                        model_channels,
                                        best_diam=None,
                                        model_invert=False,
                                        test_slice=None,
                                        diam_range=np.arange(15,51,5),
                                        ksize=25,
                                        smoothwinsize=5,
                                        hist_norm=True,
                                        kernel_size=(256,256),
                                        clip_limit=0.01,
                                        fraction_threshold=0.1,
                                        n_proc=48,
                                        bg_remove=False,
                                        use_edge=False,
                                        use_prob_weighted_score=True):

    # from .filters import var_filter
    # from .gpu import cuda_equalize_adapthist
    from tqdm import tqdm
    import pylab as plt
    from skimage.filters.rank import entropy
    import skimage.morphology as skmorph
    import skimage.exposure as skexposure
    from skimage.metrics import structural_similarity as ssim
    from scipy.stats import pearsonr
    import scipy.ndimage as ndimage
    import skimage.filters as skfilters
    # import gradient_watershed.filters as grad_filters
    from .filters import var_filter

    # test_slice = im_slice.copy()

    if best_diam is None:

        diam_score = []

        for diam in diam_range[:]:

            img = im_slice.copy()

            _, flow, style = model.cp.eval([img],
                                        channels=model_channels,
                                        batch_size=32,
                                        do_3D=False,
                                        flow_threshold=0.6,
                                        diameter=diam, # this is ok
                                        invert=model_invert) # try inverting?

            # score the content! e.g. sobel, var,
            # prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize))
            prob = flow[0][2]
            prob = 1./(1+np.exp(-prob))
            # prob_score = np.nanmean(1./(var_filter(prob, ksize=ksize) + 0.1)*prob)
            # prob_score = np.nanmean(var_filter(prob, ksize=ksize))
            # prob_score = np.median(prob) / np.std(prob)
            # prob_score = np.nanmean(entropy(prob, skmorph.disk(ksize)))
            # prob_score = np.mean(prob) / (np.std(prob)) # signal to noise ratio.
            # prob_score = np.nanmean(entropy(flow[0][1][0]/5., skmorph.disk(ksize))) + np.nanmean(entropy(flow[0][1][1]/5., skmorph.disk(ksize)))
            
            if use_prob_weighted_score:
                prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)*prob+var_filter(flow[0][1][1], ksize=ksize)*prob)  
            else:
                prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize))
            # prob_score = np.nanmean(grad_filters.var_filter(flow[0][1][0], ksize=ksize) + grad_filters.var_filter(flow[0][1][1], ksize=ksize)) # doesn't work weel
            # prob_score = ssim(img, prob) # will this  better select?
            # prob_score = np.abs(pearsonr(ndimage.uniform_filter(img, size=ksize).ravel(), prob.ravel())[0])
            # prob_score = np.nanmedian(var_filter(flow[0][1][0], ksize=ksize))+np.nanmedian(var_filter(flow[0][1][1], ksize=ksize))

            # prob_score = mask[0].max()
            # prob_score = np.nanmean(flow[0][2])
            diam_score.append(prob_score)

            # plt.figure(figsize=(5,5))
            # plt.subplot(311)
            # plt.title(str(diam)+ ' '+str(prob_score))
            # plt.imshow(img[:1024,:1024])
            # plt.subplot(312)
            # plt.title(np.mean(flow[0][2]))
            # plt.imshow(flow[0][2][:1024,:1024])
            # plt.subplot(313)
            # plt.imshow(var_filter(flow[0][2], ksize=ksize)[:1024,:1024])
            # plt.show(block=False)

        diam_score = np.hstack(diam_score)

        # smooth this.
        diam_score = _ma_average(diam_score, winsize=smoothwinsize)

        # =============================================================================
        #     Compute the best. diameter for this view.
        # =============================================================================
        best_diam = diam_range[np.argmax(diam_score)]

        # plt.figure()
        # plt.plot(diam_range, diam_score, 'o-')
        # plt.show(block=False)

    else:
        diam_score = []

    # print('auto determine cell diameter: ', best_diam)

    _, flow, style = model.cp.eval([img],
                                    batch_size=32,
                                    channels=model_channels,
                                    diameter=best_diam,
                                    # model_loaded=True,
                                    invert=model_invert,
                                    compute_masks=False)

    # return flow[0][2], flow[0][1], style[0]
    (all_probs, all_flows, all_styles) = (flow[0][2], flow[0][1], style[0])

    return (diam_range, diam_score, best_diam), (all_probs, all_flows, all_styles)


def apply_cellpose_model_2D_prob(im_stack, model, 
                            model_channels, 
                            best_diam=None, 
                            use_Cellpose_auto_diameter=False,
                            model_invert=False, 
                            test_slice=None, 
                            diam_range=np.arange(15,51,5), 
                            ksize=25,
                            smoothwinsize=5,
                            hist_norm=True, 
                            kernel_size=(256,256),
                            clip_limit=0.01,
                            fraction_threshold=0.1,
                            n_proc=48,
                            bg_remove=False,
                            use_edge=False,
                            show_img_bounds = (1024,1024),
                            debug_viz=False,
                            saveplotsfolder=None,
                            use_prob_weighted_score=False):
    
    from .filters import var_filter
    from .gpu import cuda_equalize_adapthist
    from tqdm import tqdm 
    import pylab as plt 
    from skimage.filters.rank import entropy
    import skimage.morphology as skmorph
    import skimage.exposure as skexposure
    from skimage.metrics import structural_similarity as ssim
    from scipy.stats import pearsonr
    import scipy.ndimage as ndimage
    import skimage.filters as skfilters
    import os 
    # to do port cellpose's color
    import cellpose.plot
    
    if best_diam is None:
        
        if test_slice is None:
            # don't auto into the mid slice ---> instead pick the slice with most signal. 
            if len(im_stack.shape)>3:
                if use_edge:
                    signal_stack = [np.nanmean(skfilters.sobel(np.max(im_slice,axis=-1))) for im_slice in im_stack]
                else:
                    signal_stack = [np.nanmean(np.max(im_slice,axis=-1)) for im_slice in im_stack]
            else:
                if use_edge:
                    print('single channel')
                    signal_stack = [np.nanmean(skfilters.sobel(im_slice)) for im_slice in im_stack]
                else:
                    signal_stack = [np.nanmean(im_slice) for im_slice in im_stack]
            # test_slice = len(im_stack) // 2
            test_slice = np.argmax(signal_stack)
            print('test_slice', test_slice, len(im_stack))
        
        
        if use_Cellpose_auto_diameter:
            diam_score = []
            diam_range = []
            
            img = im_stack[test_slice].copy()
            if hist_norm:
                # this messes up ? 
                img = _histeq_low_contrast(img, kernel_size=kernel_size, clip_limit=clip_limit, fraction_threshold=fraction_threshold)
                
                if bg_remove:
                    if len(img.shape) > 2: 
                        img = np.dstack([_BackgroundRemoval(img[...,ch])  for ch in np.arange(img.shape[-1])])
                    else:
                        img = _BackgroundRemoval(img)
                # img = normalize(img, pmin=2, pmax=99.8, clip=True)
                img = skexposure.rescale_intensity(img)
                # print(hist_norm)
                
            else:
                if bg_remove:
                    print('bg remove')
                    if len(img.shape) > 2: 
                        print(img.shape)
                        img = np.dstack([_BackgroundRemoval(img[...,ch])  for ch in np.arange(img.shape[-1])])
                    else:
                        img = _BackgroundRemoval(img)
                    img = skexposure.rescale_intensity(img)
                
            _, _, _, best_diam = model.eval([img], 
                                            channels=model_channels, 
                                            batch_size=32,
                                            do_3D=False, 
                                            flow_threshold=0.6, # this doesn't matter. 
                                            diameter=None, # this is ok 
                                            invert=model_invert)
                                            # model_loaded=False) 
            
        else:
            diam_range = np.hstack(diam_range)
            diam_score = []
            
            for diam in diam_range[:]: 
                
                img = im_stack[test_slice].copy()
                if hist_norm:
                    # this messes up ? 
                    img = _histeq_low_contrast(img, kernel_size=kernel_size, clip_limit=clip_limit, fraction_threshold=fraction_threshold)
                    
                    if bg_remove:
                        if len(img.shape) > 2: 
                            img = np.dstack([_BackgroundRemoval(img[...,ch])  for ch in np.arange(img.shape[-1])])
                        else:
                            img = _BackgroundRemoval(img)
                    # img = normalize(img, pmin=2, pmax=99.8, clip=True)
                    img = skexposure.rescale_intensity(img)
                    # print(hist_norm)
                    
                else:
                    if bg_remove:
                        print('bg remove')
                        if len(img.shape) > 2: 
                            print(img.shape)
                            img = np.dstack([_BackgroundRemoval(img[...,ch])  for ch in np.arange(img.shape[-1])])
                        else:
                            img = _BackgroundRemoval(img)
                        img = skexposure.rescale_intensity(img)
                    
                _, flow, style = model.cp.eval([img], 
                                            channels=model_channels, 
                                            batch_size=32,
                                            do_3D=False, 
                                            flow_threshold=0.6,
                                            diameter=diam, # this is ok 
                                            invert=model_invert) # try inverting?  
            
                # score the content! e.g. sobel, var, 
                # prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize))
                prob = flow[0][2]
                prob = np.clip(prob, -88.72, 88.72)
                prob = 1./(1+np.exp(-prob))
                # prob_score = np.nanmean(1./(var_filter(prob, ksize=ksize) + 0.1)*prob)
                # prob_score = np.nanmean(var_filter(prob, ksize=ksize))
                # prob_score = np.median(prob) / np.std(prob)
                # prob_score = np.nanmean(entropy(prob, skmorph.disk(ksize)))
                # prob_score = np.mean(prob) / (np.std(prob)) # signal to noise ratio. 
                # prob_score = np.nanmean(entropy(flow[0][1][0]/5., skmorph.disk(ksize))) + np.nanmean(entropy(flow[0][1][1]/5., skmorph.disk(ksize)))
                
                if use_prob_weighted_score:
                    prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)*prob+var_filter(flow[0][1][1], ksize=ksize)*prob)  
                else:
                    prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize))
                # prob_score = ssim(img, prob) # will this  better select? 
                # prob_score = np.abs(pearsonr(ndimage.uniform_filter(img, size=ksize).ravel(), prob.ravel())[0])
                # prob_score = np.nanmedian(var_filter(flow[0][1][0], ksize=ksize))+np.nanmedian(var_filter(flow[0][1][1], ksize=ksize))
                
                # prob_score = mask[0].max()
                # prob_score = np.nanmean(flow[0][2])
                diam_score.append(prob_score)
                
                if debug_viz:
                    plt.figure(figsize=(10,10))
                    plt.subplot(411)
                    plt.title(str(diam)+ ' '+str(prob_score))
                    if show_img_bounds is not None:
                        plt.imshow(img[:show_img_bounds[0],:show_img_bounds[1]], cmap='gray')
                    else:
                        plt.imshow(img, cmap='gray')
                    plt.subplot(412)
                    plt.title(np.mean(flow[0][2]))
                    if show_img_bounds is not None:
                        plt.imshow(prob[:show_img_bounds[0],:show_img_bounds[1]], cmap='magma', vmin=0, vmax=1) # bound this
                    else:
                        plt.imshow(prob, cmap='magma_r', vmin=0, vmax=1)
                    plt.subplot(413) # show the highlighted edges. 
                    if show_img_bounds is not None:
                        plt.imshow(var_filter(flow[0][2], ksize=ksize)[:show_img_bounds[0],:show_img_bounds[1]], cmap='coolwarm')
                    else:
                        plt.imshow(var_filter(flow[0][2], ksize=ksize), cmap='coolwarm')
                    plt.subplot(414) # show the highlighted edges. 
                    if show_img_bounds is not None:
                        plt.imshow(cellpose.plot.dx_to_circ(flow[0][1])[:show_img_bounds[0],:show_img_bounds[1]])
                    else:
                        plt.imshow(cellpose.plot.dx_to_circ(flow[0][1]))
                
                    if saveplotsfolder is not None:
                        plt.savefig(os.path.join(saveplotsfolder, 
                                                 'diameter_response_%.5f.svg' %(diam)), dpi=300, bbox_inches='tight')
                    plt.show(block=False)
                
            diam_score = np.hstack(diam_score)
            
            # smooth this. 
            diam_score = _ma_average(diam_score, winsize=smoothwinsize)
                
            # =============================================================================
            #     Compute the best. diameter for this view. 
            # =============================================================================
            best_diam = diam_range[np.argmax(diam_score)]
            
            if debug_viz:
                plt.figure(figsize=(10,10))
                plt.plot(diam_range, diam_score, 'o-')
                plt.vlines(best_diam, np.min(diam_score), np.max(diam_score), color='k', linestyles='dashed')
                plt.xlabel('Cellpose Diameter [px]')
                plt.ylabel('Contrast Score')
                plt.show(block=False)
        
    else:
        diam_score = []
        
    print('auto determine cell diameter: ', best_diam)

    all_probs = []
    all_flows = []
    all_styles = []
    
    
    for zz in tqdm(np.arange(len(im_stack))):
        img = im_stack[zz].copy()
        
        if hist_norm:
            # this messes up ? 
            img = _histeq_low_contrast(img, kernel_size=kernel_size, clip_limit=clip_limit, fraction_threshold=fraction_threshold)
            # img = normalize(img, pmin=2, pmax=99.8, clip=True)
            img = skexposure.rescale_intensity(img)
            
        _, flow, style = model.cp.eval([img],
                                        batch_size=32,
                                              channels=model_channels, 
                                              diameter=best_diam,
                                              # model_loaded=True,
                                              invert=model_invert,
                                              compute_masks=False)
        
        # _, flow, style, _ = model.eval([img],
        #                                batch_size=32,
        #                                channels=model_channels, 
        #                                diameter=best_diam,
        #                                model_loaded=True,
        #                                invert=model_invert)
        
        all_probs.append(flow[0][2])
        all_flows.append(flow[0][1])
        all_styles.append(style[0])
        
    # _, all_flows, all_styles = model.cp.eval(im_stack, 
    #                                         batch_size=32,
    #                                               channels=model_channels, 
    #                                               diameter=best_diam,
    #                                               model_loaded=True,
    #                                               invert=model_invert,
    #                                               compute_masks=False)

    # all_probs = all_flows[2].copy()
    # all_flows = all_flows[1].copy()
    
    all_probs = np.array(all_probs, dtype=np.float32)
    all_flows = np.array(all_flows, dtype=np.float32)
    all_styles = np.array(all_styles, dtype=np.float32)

    
    return (diam_range, diam_score, best_diam, test_slice), (all_probs, all_flows, all_styles)


# create function to isolate the multiscales.
def apply_cellpose_model_2D_prob_multiscale(im_stack, model, 
                                        model_channels, 
                                        best_diam=None, 
                                        model_invert=False, 
                                        test_slice=None, 
                                        diam_range=np.arange(15,51,5), 
                                        ksize=25,
                                        smoothwinsize=5,
                                        peak_prominence=0.25,
                                        peak_min_distance=None,
                                        peak_height=0.1, 
                                        peak_winsize_factor = 4,
                                        hist_norm=True, 
                                        kernel_size=(256,256),
                                        clip_limit=0.01,
                                        fraction_threshold=0.1,
                                        n_proc=48,
                                        bg_remove=False,
                                        use_edge=False,
                                        show_img_bounds = (1024,1024),
                                        include_cellpose_auto_diam=True,
                                        use_prob_weighted_score=True, 
                                        saveplotsfolder=None):
        
    from .filters import var_filter
    from .gpu import cuda_equalize_adapthist
    from tqdm import tqdm 
    import pylab as plt 
    from skimage.filters.rank import entropy
    import skimage.morphology as skmorph
    import skimage.exposure as skexposure
    from skimage.metrics import structural_similarity as ssim
    from scipy.stats import pearsonr
    import scipy.ndimage as ndimage
    import skimage.filters as skfilters
    import os 
    # to do port cellpose's color
    import cellpose.plot
    from scipy.signal import find_peaks
    
    if best_diam is None:
        diam_range = np.hstack(diam_range)
        
        if test_slice is None:
            # don't auto into the mid slice ---> instead pick the slice with most signal. 
            if len(im_stack.shape)>3:
                if use_edge:
                    signal_stack = [np.nanmean(skfilters.sobel(np.max(im_slice,axis=-1))) for im_slice in im_stack]
                else:
                    signal_stack = [np.nanmean(np.max(im_slice,axis=-1)) for im_slice in im_stack]
            else:
                if use_edge:
                    print('single channel')
                    signal_stack = [np.nanmean(skfilters.sobel(im_slice)) for im_slice in im_stack]
                else:
                    signal_stack = [np.nanmean(im_slice) for im_slice in im_stack]
            # test_slice = len(im_stack) // 2
            test_slice = np.argmax(signal_stack)
            print('test_slice', test_slice, len(im_stack))
        
        diam_score = []
        
        for diam in diam_range[:]: 
            
            img = im_stack[test_slice].copy()
            if hist_norm:
                # this messes up ? 
                img = _histeq_low_contrast(img, kernel_size=kernel_size, clip_limit=clip_limit, fraction_threshold=fraction_threshold)
                
                if bg_remove:
                    if len(img.shape) > 2: 
                        img = np.dstack([_BackgroundRemoval(img[...,ch])  for ch in np.arange(img.shape[-1])])
                    else:
                        img = _BackgroundRemoval(img)
                # img = normalize(img, pmin=2, pmax=99.8, clip=True)
                img = skexposure.rescale_intensity(img)
                # print(hist_norm)
                
            else:
                if bg_remove:
                    print('bg remove')
                    if len(img.shape) > 2: 
                        print(img.shape)
                        img = np.dstack([_BackgroundRemoval(img[...,ch])  for ch in np.arange(img.shape[-1])])
                    else:
                        img = _BackgroundRemoval(img)
                    img = skexposure.rescale_intensity(img)
                
            _, flow, style = model.cp.eval([img], 
                                        channels=model_channels, 
                                        batch_size=32,
                                        do_3D=False, 
                                        flow_threshold=0.6,
                                        diameter=diam, # this is ok 
                                        invert=model_invert) # try inverting?  
        
            # score the content! e.g. sobel, var, 
            # prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize))
            prob = flow[0][2]
            prob = 1./(1+np.exp(-prob))
            # prob_score = np.nanmean(1./(var_filter(prob, ksize=ksize) + 0.1)*prob)
            # prob_score = np.nanmean(var_filter(prob, ksize=ksize))
            # prob_score = np.median(prob) / np.std(prob)
            # prob_score = np.nanmean(entropy(prob, skmorph.disk(ksize)))
            # prob_score = np.mean(prob) / (np.std(prob)) # signal to noise ratio. 
            # prob_score = np.nanmean(entropy(flow[0][1][0]/5., skmorph.disk(ksize))) + np.nanmean(entropy(flow[0][1][1]/5., skmorph.disk(ksize)))
            
            # prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize)) # doesn't work weel 
            # prob_score = np.nanmean(prob*var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize)) # doesn't work weel 
            
            if use_prob_weighted_score:
                prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)*prob+var_filter(flow[0][1][1], ksize=ksize)*prob)  
            else:
                prob_score = np.nanmean(var_filter(flow[0][1][0], ksize=ksize)+var_filter(flow[0][1][1], ksize=ksize))
            
            
            # prob_score = ssim(img, prob) # will this  better select? 
            # prob_score = np.abs(pearsonr(ndimage.uniform_filter(img, size=ksize).ravel(), prob.ravel())[0])
            # prob_score = np.nanmedian(var_filter(flow[0][1][0], ksize=ksize))+np.nanmedian(var_filter(flow[0][1][1], ksize=ksize))
            
            # prob_score = mask[0].max()
            # prob_score = np.nanmean(flow[0][2])
            diam_score.append(prob_score)
            
            plt.figure(figsize=(10,10))
            plt.subplot(411)
            plt.title(str(diam)+ ' '+str(prob_score))
            if show_img_bounds is not None:
                plt.imshow(img[:show_img_bounds[0],:show_img_bounds[1]], cmap='gray')
            else:
                plt.imshow(img, cmap='gray')
            plt.subplot(412)
            plt.title(np.mean(flow[0][2]))
            if show_img_bounds is not None:
                plt.imshow(prob[:show_img_bounds[0],:show_img_bounds[1]], cmap='magma', vmin=0, vmax=1) # bound this
            else:
                plt.imshow(prob, cmap='magma_r', vmin=0, vmax=1)
            plt.subplot(413) # show the highlighted edges. 
            if show_img_bounds is not None:
                plt.imshow(var_filter(flow[0][2], ksize=ksize)[:show_img_bounds[0],:show_img_bounds[1]], cmap='coolwarm')
            else:
                plt.imshow(var_filter(flow[0][2], ksize=ksize), cmap='coolwarm')
            plt.subplot(414) # show the highlighted edges. 
            if show_img_bounds is not None:
                plt.imshow(cellpose.plot.dx_to_circ(flow[0][1])[:show_img_bounds[0],:show_img_bounds[1]])
            else:
                plt.imshow(cellpose.plot.dx_to_circ(flow[0][1]))
            
            if saveplotsfolder is not None:
                plt.savefig(os.path.join(saveplotsfolder, 
                                         'diameter_response_%.5f.svg' %(diam)), dpi=300, bbox_inches='tight')
            plt.show(block=False)
            
        diam_score = np.hstack(diam_score)
        
        # smooth this. 
        diam_score = _ma_average(diam_score, winsize=smoothwinsize)
            
        # =============================================================================
        #     Compute the best. diameter for this view. 
        # =============================================================================
        # best_diam = diam_range[np.argmax(diam_score)]
        
        diam_score_base = _ma_average(diam_score, winsize=len(diam_score)//peak_winsize_factor)
        diam_score_ = diam_score - diam_score_base
        
        plt.figure(figsize=(5,5))
        plt.plot(diam_score_)
        plt.show(block=False)
        
        
        # pad eitheer side to handle boundary conditions.
        diam_score_pad = np.pad(diam_score_, 
                                pad_width=[smoothwinsize,smoothwinsize], 
                                mode='reflect')
        
        best_diams = find_peaks(diam_score_pad, 
                                prominence = peak_prominence,
                                distance = peak_min_distance,
                                height = peak_height)
        
        print(best_diams)
        
        if len(best_diams[0])>0:
            
            # 
            print('testing.')
            peak_pos = best_diams[0] - smoothwinsize
            peak_pos = np.clip(peak_pos, 0, len(diam_score)-1)
            # best_diams = np.hstack([np.argmax(diam_score), best_diams[0])
            best_diams = np.hstack([np.argmax(diam_score), peak_pos])                        
            # best_diams = diam_range[best_diams[0]] # grab the range
            best_diams = diam_range[best_diams] 
            best_diams = np.unique(best_diams)
            
            # best_diams = []
            
            if include_cellpose_auto_diam:
                _, _, _, best_cp_diam = model.eval([img], 
                                                    channels=model_channels, 
                                                    batch_size=32,
                                                    do_3D=False, 
                                                    flow_threshold=0.6,
                                                    diameter=None, # this is ok 
                                                    invert=model_invert) 
                print(best_cp_diam)
                best_diams = np.hstack([best_diams, best_cp_diam]) # this is in actual dimension space. 
        else:
            best_diams = [diam_range[np.argmax(diam_score)]]


        plt.figure()
        plt.plot(diam_range, diam_score, 'o-')
        plt.show(block=False)
        
    else:
        diam_score = []
        
    print('auto determine cell diameter: ', best_diams)

    all_probs_ms = []
    all_flows_ms = []
    all_styles_ms = []
    
    for best_diam in best_diams:

        all_probs = []
        all_flows = []
        all_styles = []
        
        for zz in tqdm(np.arange(len(im_stack))):
            img = im_stack[zz].copy()
            
            if hist_norm:
                # this messes up ? 
                img = _histeq_low_contrast(img, kernel_size=kernel_size, clip_limit=clip_limit, fraction_threshold=fraction_threshold)
                # img = normalize(img, pmin=2, pmax=99.8, clip=True)
                img = skexposure.rescale_intensity(img)
                
            _, flow, style = model.cp.eval([img],
                                            batch_size=32,
                                                  channels=model_channels, 
                                                  diameter=best_diam,
                                                  # model_loaded=True,
                                                  invert=model_invert,
                                                  compute_masks=False)
            
            # _, flow, style, _ = model.eval([img],
            #                                batch_size=32,
            #                                channels=model_channels, 
            #                                diameter=best_diam,
            #                                model_loaded=True,
            #                                invert=model_invert)
            all_probs.append(flow[0][2])
            all_flows.append(flow[0][1])
            all_styles.append(style[0])
            
        # _, all_flows, all_styles = model.cp.eval(im_stack, 
        #                                         batch_size=32,
        #                                               channels=model_channels, 
        #                                               diameter=best_diam,
        #                                               model_loaded=True,
        #                                               invert=model_invert,
        #                                               compute_masks=False)
    
        # all_probs = all_flows[2].copy()
        # all_flows = all_flows[1].copy()
        all_probs = np.array(all_probs, dtype=np.float32)
        all_flows = np.array(all_flows, dtype=np.float32)
        all_styles = np.array(all_styles, dtype=np.float32)
        
        all_probs_ms.append(all_probs)
        all_flows_ms.append(all_flows)
        all_styles_ms.append(all_styles)
        
    all_probs_ms = np.array(all_probs_ms)
    all_flows_ms = np.array(all_flows_ms)
    all_styles_ms = np.array(all_styles_ms)
    

    return (diam_range, diam_score, best_diams, test_slice), (all_probs_ms, all_flows_ms, all_styles_ms)


