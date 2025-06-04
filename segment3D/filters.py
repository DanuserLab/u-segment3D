# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:37:54 2023

@author: fyz11

helper filters. 
"""

import numpy as np 


def fill_array(im, thresh=0, fill_vals=0, method='constant'):
    
    import numpy as np 
    
    out = im.copy()
    
    if method == 'constant':
        out[im<=thresh] = fill_vals
    if method == 'median':
        out[im<=thresh] = np.nanmedian(im[im>thresh])
    if method == 'mean':
        out[im<=thresh] = np.nanmean(im[im>thresh])
        
    return out

def imadjust(vol, p1, p2): 
    import numpy as np 
    from skimage.exposure import rescale_intensity
    # this is based on contrast stretching and is used by many of the biological image processing algorithms.
    p1_, p2_ = np.percentile(vol, (p1,p2))
    vol_rescale = rescale_intensity(vol, in_range=(p1_,p2_))
    return vol_rescale

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)
    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )
    if clip:
        x = np.clip(x,0,1)
    return x

def normalize(x, pmin=2, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):
    
    """
    Anisotropic diffusion in 2D 

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """
    
    # import skimage.filters as flt
    import scipy.ndimage.filters as flt
    import warnings 
    import skimage.io as io
    import matplotlib
    import numpy as np 
    
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in np.arange(1,niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        if 0<sigma:
            deltaSf=flt.gaussian_filter(deltaS,sigma);
            deltaEf=flt.gaussian_filter(deltaE,sigma);
        else: 
            deltaSf=deltaS;
            deltaEf=deltaE;
            
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
            gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return imgout


def meijering_ridge_filter_2D_multiprocess(stack, sigmas, low_contrast_percentile=0.05, black_ridges=False, n_cpu=4):
    
    import multiprocess as mp
    import skimage.filters as skfilters 
    import skimage.exposure as skexposure
    import numpy as np 
    
    stack_in = normalize(stack, pmin=0, pmax=100, clip=True) # ensure 0-1
    
    def _run_filter(slice_id):
        
        im_slice = stack_in[slice_id]
        if skexposure.is_low_contrast(im_slice, fraction_threshold=low_contrast_percentile):
            im_slice = np.zeros_like(im_slice)
        else:
            im_slice = skfilters.meijering(im_slice, 
                                           sigmas=sigmas, 
                                           black_ridges=black_ridges)
        return im_slice
        
    with mp.Pool(n_cpu) as pool: # potentially thread is faster? 
        res = pool.map(_run_filter, range(0, len(stack_in)))

    return np.array(res)


def relabel_slices(labelled, bg_label=0):
    
    import numpy as np 
    max_ID = 0 
    labelled_ = []
    for lab in labelled:
        lab_slice = lab.copy()
        lab_slice[lab_slice>bg_label] = lab[lab>bg_label] + max_ID # only update the foreground! 
        labelled_.append(lab_slice)
        max_ID += np.max(lab_slice)
        
    labelled_ = np.array(labelled_)
    
    return labelled_


def connected_component_slices(labelled,bg_label=0):
    
    import skimage.measure as skmeasure
    import numpy as np 
    
    max_ID = 0 
    labelled_ = []
    for lab in labelled:
        lab_slice = lab.copy()
        lab_slice_label = skmeasure.label(lab_slice)
        lab_slice[lab_slice>bg_label] = lab_slice_label[lab_slice_label>0] + max_ID # only update the foreground! 
        labelled_.append(lab_slice)
        max_ID += np.max(lab_slice_label)
        
    labelled_ = np.array(labelled_)
    
    return labelled_


def filter_2d_label_slices(labelled,bg_label=0, minsize=10):
    
    import skimage.measure as skmeasure 
    import numpy as np 
    
    labelled_ = []
    for lab in labelled:
        lab_slice = lab.copy()
        # now check every existing label 
        uniq_labels = np.setdiff1d(np.unique(lab_slice), bg_label)
        
        max_ID = 0
        # iterate and correct all the labels. 
        label_slice_new = np.zeros_like(lab_slice)
        
        for ll in uniq_labels:
            binary = lab_slice == ll
            lab_binary = skmeasure.label(binary>0)
            uniq_label_slices = np.setdiff1d(np.unique(lab_binary),0)
            
            for lab in uniq_label_slices:
                lab_mask = lab_binary==lab
                if np.sum(lab_mask) > minsize:
                    label_slice_new[lab_binary==lab] = max_ID + 1
                    max_ID += 1
                
        labelled_.append(label_slice_new)
      
    labelled_ = np.array(labelled_)
    
    return labelled_


def detect_and_relabel_multi_component_labels(labels, min_size=100):
    
    import skimage.measure as skmeasure
    import skimage.morphology as skmorph
    
    labels_filt = np.zeros_like(labels)
    
    max_id = 1

    regionprops = skmeasure.regionprops(labels)
    
    for cc, re in enumerate(regionprops[:]):
        
        bbox = re.bbox
        coords = re.coords
        
        x1,y1,z1 = bbox[:3]
        bbox_size = np.hstack(bbox[3:]) - np.hstack(bbox[:3])
        
        mask = np.zeros(bbox_size, dtype=bool)
        mask[coords[:,0]-x1,
             coords[:,1]-y1,
             coords[:,2]-z1] = 1
        
        mask = skmorph.remove_small_objects(mask, min_size=min_size)
        label_mask = skmeasure.label(mask>0)
        
        label_ids = np.setdiff1d(np.unique(label_mask),0)
        
        for ll in label_ids:
            coords_ll = np.argwhere(label_mask==ll)
            coords_ll = coords_ll + np.hstack(bbox[:3])
            
            labels_filt[coords_ll[:,0], 
                                coords_ll[:,1], 
                                coords_ll[:,2]] = max_id
            max_id += 1
    
    return labels_filt


def filter_segmentations_axis(labels, window=3, min_count=1):
    """ according to the persistence along 1 direction. - this works and is relatively fast. 
    """
    import skimage.measure as skmeasure 
    
    labels_filt = []
    N = len(labels)
    offset = window//2
    
    labels_pad = np.pad(labels, [[offset,offset], [0,0], [0,0]], mode='reflect')
    
    for ss_ii in np.arange(N):
        # base
        ss = labels_pad[ss_ii+offset].copy()
        ss_copy = ss.copy()
        
        # iterate over the unique regions. 
        uniq_reg = np.setdiff1d(np.unique(ss),0)
        
        for reg in uniq_reg:
            mask = ss == reg
            
            checks = np.zeros(2*offset)
            cnt = 0
            for ii in np.arange(0, 2*offset+1):
                # now we are going to apply. 
                if ii != offset:
                    # print(ii)
                    mask_ii = labels_pad[ss_ii+cnt, mask>0].copy()
                    
                    # print(np.sum(mask_ii>0))
                    if np.sum(mask_ii>0) >= min_count:
                        checks[cnt] += 1 # append true. 
                    
                    cnt+=1
                # else:
                #     checks[cnt] += 1
                #     cnt+=1
                    
                    
            # print(checks)
            # if np.max(checks) < offset: # less than half i.e. not persistent # so this is wrong!. --- what i was going to do is the below. 
            # i think we do need to check continguousness!. 
            
            # check contiguity! 
            contigs = skmeasure.label(checks)
            
            # print(contigs)
            
            if contigs.max() > 0:
                uniq_contig = np.setdiff1d(np.unique(contigs), 0)
                lengths = [np.sum(contigs==cc) for cc in uniq_contig]
                mask = contigs == uniq_contig[np.argmax(lengths)]
                
                # print(mask)
                # print(contigs)
                # print(np.sum(mask)>= window //2 + 1)
                # print('---')
                
                # check the length of mask
                if np.sum(mask)>= window //2 + 1:
                    # if sufficiently long. 
                    if mask[offset-1] == 0 and mask[offset] == 0: # that is doesn't cover either of this !. 
                        ss_copy[ss==reg] = 0
                else:
                    # missing clause! 
                    ss_copy[ss==reg] = 0
            else:
                ss_copy[ss==reg] = 0 # zero it
            # if np.sum(checks) < len(checks): 
            #     ss_copy[ss==reg] = 0 # zero it
                
        labels_filt.append(ss_copy)
        
    labels_filt = np.array(labels_filt)
    
    return labels_filt



def filter_segmentations_axis_IoU(labels, window=3, iou_var = 0.1):
    """ according to the persistence along 1 direction of the IoU with respect to the middle reference shape.  
    """
    
    labels_filt = []
    N = len(labels)
    offset = window//2
    
    labels_pad = np.pad(labels, [[offset,offset], [0,0], [0,0]], mode='reflect')
    
    for ss_ii in np.arange(N):
        # base
        ss = labels_pad[ss_ii+offset].copy()
        ss_copy = ss.copy()
        
        # iterate over the unique regions. 
        uniq_reg = np.setdiff1d(np.unique(ss),0)
        
        for reg in uniq_reg:
            mask = ss == reg
            
            iou_checks = np.zeros(2*offset)
            cnt = 0
            for ii in np.arange(0, 2*offset+1):
                # now we are going to apply. 
                if ii != offset:
                    # print(ii)
                    mask_ii = labels_pad[ss_ii+cnt, mask>0].copy()
                    
                    if mask_ii.max()>0:
                        uniq_regions_mask_ii = np.setdiff1d(np.unique(mask_ii),0)
                        largest = uniq_regions_mask_ii[np.argmax([np.sum(labels_pad[ss_ii+cnt]==rr) for rr in uniq_regions_mask_ii])]
                        largest_mask = labels_pad[ss_ii+cnt]==largest
                        # compute the IoU overlap. 
                        overlap = np.sum(np.logical_and(largest_mask, mask))
                        iou = np.sum(np.logical_and(largest_mask, mask)) / (np.sum(mask)+np.sum(largest_mask) - overlap)
                        iou_checks[cnt] = iou
                    else:
                        iou_checks[cnt] = 0 
                    
                    # # print(np.sum(mask_ii>0)>0)
                    # if np.sum(mask_ii>0) >= min_count:
                    #     checks[cnt] += 1 # append true. 
                    cnt+=1
                # else:
                    
            # print(checks)
            # if np.max(checks) < offset: # less than half i.e. not persistent # so this is wrong!. --- what i was going to do is the below. 
            # print(iou_checks, np.std(iou_checks))
            if np.std(iou_checks) > iou_var: 
                ss_copy[ss==reg] = 0 # zero it
                
        labels_filt.append(ss_copy)
        
    labels_filt = np.array(labels_filt)
    
    return labels_filt



# potentially extend this to handle anistropic! 
def smooth_vol(vol_binary, ds=4, smooth=5):
    
    from skimage.filters import gaussian
    from scipy.ndimage import gaussian_filter
    import skimage.transform as sktform
    import numpy as np 
    
    small = sktform.resize(vol_binary, np.array(vol_binary.shape)//ds, preserve_range=True)
    small = gaussian_filter(small, sigma=smooth)
    
    return sktform.resize(small, np.array(vol_binary.shape), preserve_range=True)


def smooth_vol_anisotropic(vol_binary, ds=4, smooth=[5,5,5]):
    
    from skimage.filters import gaussian
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    import skimage.transform as sktform
    import numpy as np 
    
    small = sktform.resize(vol_binary, np.array(vol_binary.shape)//ds, preserve_range=True)
    # small = gaussian_filter(small, sigma=smooth)
    for axis in np.arange(len(smooth)):
        small = gaussian_filter1d(small, sigma=smooth[axis], axis=axis) # apply separable filtering 
    
    return sktform.resize(small, np.array(vol_binary.shape), preserve_range=True)


# =============================================================================
# for cleaning labels
# =============================================================================
def remove_small_labels(labels, min_size=64):
    
    import skimage.measure as skmeasure
    
    if labels.max()>0:
        uniq_regions = np.setdiff1d(np.unique(labels),0)
        props = skmeasure.regionprops(labels)
    
        areas = np.hstack([re.area for re in props])
        # print(areas)
        regions_remove = uniq_regions[areas<min_size]
        
        labels_new = labels.copy()
        for reg in regions_remove:
            labels_new[labels==reg] = 0 # set to background
        
        return labels_new
    else:
        
        return labels
    
    
def intensity_filter_labels(labels, intensity_img, threshold=None, auto_thresh_method='Otsu'):
    
    import skimage.measure as skmeasure 
    import skimage.filters as skfilters

    labels_filt = labels.copy()
    regionprops = skmeasure.regionprops(labels)
    
    all_region_I = [] 
    all_region_I_inds = []
    
    for ind, reg in enumerate(regionprops):
        coords = reg.coords
        coords_I = np.nanmean(intensity_img[coords[:,0], 
                                            coords[:,1],
                                            coords[:,2]])
        all_region_I.append(coords_I)
        all_region_I_inds.append(ind)
        
    if len(all_region_I) > 0:
        all_region_I = np.hstack(all_region_I)
        all_region_I_inds = np.hstack(all_region_I_inds)
        
        if threshold is None:
            if auto_thresh_method == 'Otsu':
                threshold = skfilters.threshold_otsu(all_region_I)
            elif auto_thresh_method == 'Mean':
                threshold = np.nanmean(all_region_I)
            elif auto_thresh_method == 'Median':
                threshold = np.nanmedian(all_region_I)
            else:
                threshold = skfilters.threshold_otsu(all_region_I)
        
        remove_inds = all_region_I_inds[all_region_I<threshold]
        
        for rr in remove_inds:
            coords = regionprops[rr].coords
            labels_filt[coords[:,0], coords[:,1], coords[:,2]] = 0 # zero out
        
    return labels_filt
            
        
def expand_masks(label_seeds, binary, dist_tform=None):
    
    import skimage.measure as skmeasure 
    from skimage.segmentation import watershed
    
    if dist_tform is None:
        
        from .flows import distance_transform_labels_fast 
        dist_tform = distance_transform_labels_fast(skmeasure.label(binary, connectivity=2))
    
    # use the initial labels as seed
    seeds = label_seeds * binary
    
    labels_refine = watershed(-dist_tform, seeds, mask=binary) # this fills in the binary as much as possible. 
    
    # the remainder we can label otherwise. 
    remainder = np.logical_and(binary>0, labels_refine==0)
    remainder_label = skmeasure.label(remainder)
    remainder_label[remainder>0] = remainder_label[remainder>0] + label_seeds.max()
    
    
    return labels_refine


def expand_masks2D(label_seeds, binary, dist_tform=None):
    
    import skimage.measure as skmeasure 
    from skimage.segmentation import watershed
    
    if dist_tform is None:
        
        from .flows import distance_transform_labels_fast 
        dist_tform = distance_transform_labels_fast(skmeasure.label(binary, connectivity=2)) # this is running 2D slice by slice!. 
    
    # use the initial labels as seed
    seeds = label_seeds * binary # 
    
    labels_refine = []
    
    for dd in np.arange(len(binary)):
        labels_refine_slice = watershed(-dist_tform[dd], seeds[dd], mask=binary[dd]) # this fills in the binary as much as possible. 
    
        # the remainder we can label otherwise. 
        remainder = np.logical_and(binary[dd]>0, labels_refine_slice==0)
        remainder_label = skmeasure.label(remainder)
        remainder_label[remainder>0] = remainder_label[remainder>0] + label_seeds.max()
        
        labels_refine_slice[remainder>0] = remainder_label[remainder>0].copy()
        labels_refine.append(labels_refine_slice)
    labels_refine = np.array(labels_refine)
    
    return labels_refine
    

def remove_eccentric_shapes(labels, min_size=20, stretch_cutoff=15):

    ##### get the objects 
    import numpy as np 
    
    labels_filter = labels.copy()

    obj_area = []
    obj_ecc = [] 
    
    uniq_labs_3D = np.setdiff1d(np.unique(labels), 0)
    
    for lab in uniq_labs_3D[:]:
        mask = labels==lab 
        obj_area.append(np.nansum(mask))
        
        pts_mask = np.vstack(np.argwhere(mask>0))
        
        if len(pts_mask) > min_size:
            cov = np.cov((pts_mask-pts_mask.mean(axis=0)[None,:]).T)
            eigs = np.linalg.eigvalsh(cov)
            
            stretch_ratio = np.max(np.max(np.abs(eigs))/np.abs(eigs))
            
            if stretch_ratio > stretch_cutoff :
                labels_filter[mask] = 0
            
            obj_ecc.append(stretch_ratio)
        else:
            obj_ecc.append(np.nan) # no obj_ecc
            labels_filter[mask] = 0
            
    return labels_filter, (obj_area, obj_ecc)


# =============================================================================
# for fusion
# =============================================================================
def entropy_mean(arrays, ksize=3,alpha = 0.5, eps=1e-20):
        
    import skimage.filters.rank as skfilters_rank #import entropy 
    import skimage.morphology as skmorph
    
    ents = np.array([1./(skfilters_rank.entropy(array, selem=skmorph.ball(ksize)) + alpha) for array in arrays])
    
    v = np.sum(ents*arrays, axis=0) / (ents.sum(axis=0)+eps)
    
    return v


def var_filter(img, ksize):
    import scipy.ndimage as ndimage
    import numpy as np 
    
    win_mean = ndimage.uniform_filter(img,ksize)
    win_sqr_mean = ndimage.uniform_filter(img**2, ksize)
    win_var = win_sqr_mean - win_mean**2
    
    return win_var


def fast_dog_filter(img, ksize, norm=True):
    
    import scipy.ndimage as ndimage
    import numpy as np 
    
    out = ndimage.uniform_filter(img, size=1) - ndimage.uniform_filter(img, size=ksize)
    
    if norm:
        out = (out - np.mean(out)) / np.std(out)
    
    return out 

def var_combine(arrays, ksize=3, alpha = 1., eps=1e-20):
        
    import skimage.filters.rank as skfilters_rank #import entropy 
    import skimage.morphology as skmorph
    
    weights = np.array([1./(var_filter(array, ksize=ksize) + alpha) for array in arrays])
    
    v = np.sum(weights*arrays, axis=0) / (weights.sum(axis=0)+eps)
    
    return v

def dog_combine(arrays, ksize=3, eps=1e-20):
        
    # import skimage.filters.rank as skfilters_rank #import entropy 
    # import skimage.morphology as skmorph
    import numpy as np 
    
    weights = np.array([np.clip(fast_dog_filter(array, ksize=ksize, norm=True), 0, 4) for array in arrays])
    
    v = np.sum(weights*arrays, axis=0) / (weights.sum(axis=0)+eps)
    
    return v


# =============================================================================
# 3D guided filter
# =============================================================================

def guidedfilter(I, G, radius, eps):
    """ n-dimensional guided filter. 
    
    References:
        K.He, J.Sun, and X.Tang. Guided Image Filtering. TPAMI'12.
    """
    import numpy as np 
    import scipy.ndimage as ndimage 
    
    
    # step 1
    meanI  = ndimage.uniform_filter(G, size=radius)
    meanp  = ndimage.uniform_filter(I, size=radius)
    corrI  = ndimage.uniform_filter(G*G, size=radius)
    corrIp = ndimage.uniform_filter(I*G, size=radius)
    
    # step 2
    varI   = corrI - meanI * meanI
    covIp  = corrIp - meanI * meanp
    
    # step 3
    a      = covIp / (varI + eps)
    b      = meanp - a * meanI
    # step 4
    meana  = ndimage.uniform_filter(a, size=eps)
    meanb  = ndimage.uniform_filter(b, size=eps)
    # step 5
    q = meana * G + meanb

    return q

    
def largest_component_vol(vol_binary, connectivity=1):
    
    from skimage.measure import label, regionprops
    import numpy as np 
    
    vol_binary_labelled = label(vol_binary, connectivity=connectivity)
    # largest component.
    vol_binary_props = regionprops(vol_binary_labelled)
    vol_binary_vols = [re.area for re in vol_binary_props]
    vol_binary = vol_binary_labelled == (np.unique(vol_binary_labelled)[1:][np.argmax(vol_binary_vols)])
    
    return vol_binary


def largest_component_vol_labels(vol_labels, connectivity=1):
    
    import numpy as np 
    
    # now do a round well each only keeps the largest component. 
    uniq_labels = np.setdiff1d(np.unique(vol_labels), 0)
    vol_labels_new = np.zeros_like(vol_labels) # must be like this!. 

    for lab in uniq_labels:
        mask = vol_labels == lab 
        mask = largest_component_vol(mask, connectivity=connectivity)
        vol_labels_new[mask>0] = lab # put this. 
    
    return vol_labels_new

def largest_component_vol_labels_fast(vol_labels, connectivity=1):
    
    ### filter for largest component for each component and keep only the largest connected component. 
    # put this in a module !. 
    import cc3d
    import numpy as np 
    import skimage.measure as skmeasure
    # from tqdm import tqdm 
    
    vol_labels_clean = np.zeros_like(vol_labels)
    
    # region_labels = np.setdiff1d(np.unique(vol_labels), 0)
    regions = skmeasure.regionprops(vol_labels) # supports bbox. 
    print(len(regions))
    
    for reg_ii, reg in enumerate(regions):
        
        bbox = reg.bbox
        bbox_binary = reg.image
        if connectivity==1:
            bbox_label = cc3d.connected_components(bbox_binary, connectivity=18)
            bbox_label, N = cc3d.largest_k(bbox_label, k=1, connectivity=18, return_N=True)
        if connectivity==2:
            bbox_label = cc3d.connected_components(bbox_binary, connectivity=26)
            bbox_label, N = cc3d.largest_k(bbox_label, k=1, connectivity=26, return_N=True)
        
        coords = np.argwhere(bbox_label>0) 
        
        if len(coords)>0:
            coords = coords + bbox[:3]
        
            vol_labels_clean[coords[:,0], 
                             coords[:,1], 
                             coords[:,2]] = reg_ii + 1 # relable. 
            
    return vol_labels_clean
    
    

def _distance_to_heat_affinity_matrix(Dmatrix, gamma=None):
    r""" Convert any distance matrix to an affinity matrix by applying a heat kernel.

    .. math:: 
        A = \exp^{\left(\frac{-D^2}{2\sigma^2}\right)}

    where :math:`sigma` is set as the mean distance of :math:`D` or :math:`\gamma` if provided.
    
    Parameters
    ----------
    Dmatrix : (N,N) sparse array
        a scipy.sparse input distance matrix
    gamma : scalar
        the normalisation scale factor of distances

    Returns 
    -------
    A : (N,N) sparse array
        a scipy.sparse output affinity distance matrix

    """
    import numpy as np 
    # import igl
    import scipy.sparse as spsparse

    l = Dmatrix.shape[0]
    A = Dmatrix.copy()
    if gamma is None:
        sigma_D = np.mean(A.data)
    else:
        sigma_D = gamma
    den_D = 2 * (sigma_D ** 2)
    np.exp( -A.data**2/den_D, out=A.data )
    A = A + spsparse.diags(np.ones(l), 0)  # diagonal is 1 by definition. 

    return A.tocsr() # this should give faster? 


def _distance_to_laplace_affinity_matrix(Dmatrix, gamma=None):
    r""" Convert any distance matrix to an affinity matrix by applying a heat kernel.

    .. math:: 
        A = \exp^{\left(\frac{-|D|}{\sigma}\right)}

    where :math:`sigma` is set as the mean distance of :math:`D` or :math:`\gamma` if provided.
    
    Parameters
    ----------
    Dmatrix : (N,N) sparse array
        a scipy.sparse input distance matrix
    gamma : scalar
        the normalisation scale factor of distances

    Returns 
    -------
    A : (N,N) sparse array
        a scipy.sparse output affinity distance matrix

    """
    import numpy as np 
    # import igl
    import scipy.sparse as spsparse

    l = Dmatrix.shape[0]
    A = Dmatrix.copy()
    if gamma is None:
        sigma_D = np.mean(A.data)
    else:
        sigma_D = gamma
    den_D = sigma_D 
    np.exp( -np.abs(A.data)/den_D, out=A.data )
    A = A + spsparse.diags(np.ones(l), 0)  # diagonal is 1 by definition. 

    return A.tocsr() # this should give faster? 


def diffuse_labels2D(labels_in, guide, clamp=0.99, n_iter=10, noprogress=True, alpha=0.8, affinity_type='heat',reinitialize_freq=None):
    """ 
    
    Parameters
    ----------
    labels_in : TYPE
        DESCRIPTION.
    guide : TYPE
        DESCRIPTION.
    clamp : TYPE, optional
        DESCRIPTION. The default is 0.99.
    n_iter : TYPE, optional
        DESCRIPTION. The default is 10.
    noprogress : TYPE, optional
        DESCRIPTION. The default is True.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.8.

    Returns
    -------
    z_label : TYPE
        DESCRIPTION.

    """
    # we need to test this function ! the original was infact wrong? 
    from sklearn.feature_extraction.image import img_to_graph, grid_to_graph
    from tqdm import tqdm
    import skimage.segmentation as sksegmentation 
    
    # relabel labels_in 
    labels_in_relabel, fwd, bwd = sksegmentation.relabel_sequential(labels_in) # fwd, original -> new, bwd, new->original
    
    graph = img_to_graph(guide) # use gradients
    if affinity_type=='heat':
        affinity = _distance_to_heat_affinity_matrix(graph, gamma=None)
    elif affinity_type =='laplace':
        affinity = _distance_to_laplace_affinity_matrix(graph, gamma=None)
    else:
        print('not valid')
    # normalize this.... 

    graph_laplacian=grid_to_graph(guide.shape[0], guide.shape[1])
    if affinity_type=='heat':
        affinity_laplacian = _distance_to_heat_affinity_matrix(graph_laplacian*1., gamma=None)
    elif affinity_type =='laplace':
        affinity_laplacian = _distance_to_laplace_affinity_matrix(graph_laplacian*1., gamma=None)
    else:
        print('not valid')
    
    affinity = alpha * affinity + (1.-alpha) * affinity_laplacian # take an average. 
    
    n_labels = np.max(labels_in_relabel)+1 # include background!. 
    
    labels = np.zeros((np.prod(labels_in_relabel.shape[:2]), n_labels), 
                          dtype=np.float32)    

    labels[np.arange(len(labels_in_relabel.ravel())), labels_in_relabel.ravel()] = 1 # set all the labels 
    
    # # diffuse on this.... with label propagation.
    # alpha_prop = clamp
    # base_matrix = (1.-alpha_prop)*labels
    # # above looks wrong... 
    # alpha_prop = 1.-clamp
    # base_matrix = (1.-alpha_prop)*labels
    
    init_matrix = np.zeros_like(labels) # let this be the new. 
    
    for ii in tqdm(np.arange(n_iter), disable=noprogress):
        # init_matrix = affinity.dot(init_matrix) + base_matrix
        init_matrix = (1.-clamp)*affinity.dot(init_matrix) + clamp*labels # this is the correct equation.
        
        # print(np.mean(init_matrix))
        
        # # at some point this can saturate... 
        # if reinitialize_freq is not None:
            
        #     init_matrix = init_matrix / np.nanmean(init_matrix)
        #     # print('reinitialize_freq')
        #     # z = np.nansum(init_matrix, axis=1)
        #     # z[z==0] += 1 # Avoid division by 0
        #     # z = ((init_matrix.T)/z).T
        #     # init_matrix = z.copy()
        #     # # if np.mod(ii+1, reinitialize_freq) == 0:
        #     # #     z = np.nansum(init_matrix, axis=1) # sum all counts.
        #     # #     z[z==0] += 1 # Avoid division by 0
        #     # #     z = ((init_matrix.T)/z).T # renormalize. 
        #     # #     z_label = np.argmax(z, axis=1)
                
        #     # #     init_matrix = np.zeros_like(labels)
        #     # #     init_matrix[np.arange(len(z_label.ravel())), z_label.ravel()] = 1
        #     # #     # init_matrix = np.zeros_like(labels)
                
    z = np.nansum(init_matrix, axis=1)
    z[z==0] += 1 # Avoid division by 0
    z = ((init_matrix.T)/z).T
    z_label = np.argmax(z, axis=1)
    z_label = z_label.reshape(labels_in_relabel.shape)
    
    # map back. 
    z_label = bwd[z_label]
    return z_label


def diffuse_labels3D(labels_in, guide, clamp=0.99, n_iter=10, noprogress=True, alpha=0.8, affinity_type='heat',reinitialize_freq=None):
    """ 
    
    Parameters
    ----------
    labels_in : TYPE
        DESCRIPTION.
    guide : TYPE
        DESCRIPTION.
    clamp : TYPE, optional
        DESCRIPTION. The default is 0.99.
    n_iter : TYPE, optional
        DESCRIPTION. The default is 10.
    noprogress : TYPE, optional
        DESCRIPTION. The default is True.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.8.

    Returns
    -------
    z_label : TYPE
        DESCRIPTION.

    """
    # we need to test this function ! the original was infact wrong? 
    from sklearn.feature_extraction.image import img_to_graph, grid_to_graph
    from tqdm import tqdm
    import skimage.segmentation as sksegmentation 
    
    # relabel labels_in 
    labels_in_relabel, fwd, bwd = sksegmentation.relabel_sequential(labels_in) # fwd, original -> new, bwd, new->original
    
    graph = img_to_graph(guide) # use gradients
    if affinity_type=='heat':
        affinity = _distance_to_heat_affinity_matrix(graph, gamma=None)
    elif affinity_type =='laplace':
        affinity = _distance_to_laplace_affinity_matrix(graph, gamma=None)
    else:
        print('not valid')
    # normalize this.... 

    graph_laplacian=grid_to_graph(guide.shape[0], guide.shape[1], guide.shape[2])
    if affinity_type=='heat':
        affinity_laplacian = _distance_to_heat_affinity_matrix(graph_laplacian*1., gamma=None)
    elif affinity_type =='laplace':
        affinity_laplacian = _distance_to_laplace_affinity_matrix(graph_laplacian*1., gamma=None)
    else:
        print('not valid')
    
    affinity = alpha * affinity + (1.-alpha) * affinity_laplacian # take an average. 
    
    n_labels = np.max(labels_in_relabel)+1 # include background!. 
    
    labels = np.zeros((np.prod(labels_in_relabel.shape[:3]), n_labels), 
                          dtype=np.float32)    

    labels[np.arange(len(labels_in_relabel.ravel())), labels_in_relabel.ravel()] = 1 # set all the labels 
    
    # # diffuse on this.... with label propagation.
    # alpha_prop = clamp
    # base_matrix = (1.-alpha_prop)*labels
    # # above looks wrong... 
    # alpha_prop = 1.-clamp
    # base_matrix = (1.-alpha_prop)*labels
    
    init_matrix = np.zeros_like(labels) # let this be the new. 
    
    for ii in tqdm(np.arange(n_iter), disable=noprogress):
        # init_matrix = affinity.dot(init_matrix) + base_matrix
        init_matrix = (1.-clamp)*affinity.dot(init_matrix) + clamp*labels # this is the correct equation.
        
        # print(np.mean(init_matrix))
        
        # at some point this can saturate... 
        if reinitialize_freq is not None:
            
            init_matrix = init_matrix / np.nanmean(init_matrix)
            # print('reinitialize_freq')
            # z = np.nansum(init_matrix, axis=1)
            # z[z==0] += 1 # Avoid division by 0
            # z = ((init_matrix.T)/z).T
            # init_matrix = z.copy()
            # # if np.mod(ii+1, reinitialize_freq) == 0:
            # #     z = np.nansum(init_matrix, axis=1) # sum all counts.
            # #     z[z==0] += 1 # Avoid division by 0
            # #     z = ((init_matrix.T)/z).T # renormalize. 
            # #     z_label = np.argmax(z, axis=1)
                
            # #     init_matrix = np.zeros_like(labels)
            # #     init_matrix[np.arange(len(z_label.ravel())), z_label.ravel()] = 1
            # #     # init_matrix = np.zeros_like(labels)
                
    z = np.nansum(init_matrix, axis=1)
    z[z==0] += 1 # Avoid division by 0
    z = ((init_matrix.T)/z).T
    z_label = np.argmax(z, axis=1)
    z_label = z_label.reshape(labels_in_relabel.shape)
    
    # map back. 
    z_label = bwd[z_label]
    return z_label
