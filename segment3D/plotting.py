# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 00:26:42 2023

@author: fyz11
"""

import pylab as plt 

def set_axes_equal(ax: plt.Axes):
    
    import numpy as np 
    
    limits = np.array([
        ax.get_xlim3d(), 
        ax.get_ylim3d(), 
        ax.get_zlim3d(), 
        ])
    origin = np.mean(limits, axis=1)
    radius = 0.5*np.max(np.abs(limits[:,1] - limits[:,0]))
    _set_axes_radius(ax, origin, radius)
    
    return [] 

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin 
    ax.set_xlim3d([x - radius, x+radius])
    ax.set_ylim3d([y - radius, y+radius])
    ax.set_zlim3d([z - radius, z+radius])
    
    return [] 


# this is good. 
def relabel_ground_truth_labels(mask, connectivity=2, minsize=10, reorder=True):
    
    import skimage.measure as skmeasure 
    import skimage.morphology as skmorph
    import numpy as np 
    # import skimage.segmentation as sksegmentation 
    
    # get the labelled regions. 
    uniq_regions = np.setdiff1d(np.unique(mask),0)
    masks_out = np.zeros_like(mask)
    
    max_label = 0 
    centroids = []
    
    for rr in uniq_regions: 
    
        region = mask == rr
        # rerun connected components. 
        region = skmorph.remove_small_objects(region, min_size=minsize)
        region_cc = skmeasure.label(region, connectivity=connectivity) # use the full connectivity 
        
        regprops = skmeasure.regionprops(region_cc)
        centers = [re.centroid for re in regprops]
        
        if len(centers)> 0:
            centroids.append(np.vstack(centers))
        
        masks_out[region_cc>0] = max_label+region_cc[region_cc>0]
        max_label = masks_out.max()
        
    centroids = np.vstack(centroids)
    
    if reorder:
        ind = np.lexsort((centroids[:,1], centroids[:,0])) # sort by x then y 
        
        masks_out_reorder = np.zeros_like(masks_out)
        
        for ii, lab in enumerate(ind):
            masks_out_reorder[masks_out==lab+1] = ii + 1
        return masks_out_reorder
    # masks_out = sksegmentation.relabel_sequential(masks_out)[0]
    else:    
        return masks_out
        
    
def color_segmentation(labels, cmapname, n_colors=16):
    
    import skimage.color as skcolor
    import numpy as np 
    import seaborn as sns 
    
    labels_color = np.uint8(255*skcolor.label2rgb(labels, 
                                              colors=sns.color_palette(cmapname, n_colors=n_colors), # spectral is a nicer colorscheme. 
                                              bg_label=0))
    
    return labels_color
    

def get_colors(inp, colormap, vmin=None, vmax=None, bg_label=None):
	import pylab as plt 
    
	norm = plt.Normalize(vmin, vmax)

	colored = colormap(norm(inp))
	if bg_label is not None:
		colored[inp==bg_label] = 0 # make these all black!


	return colored


def plot_border_tracks_2D(tracks, labels, ax, lw=1, color='magenta', samples=10):
    
    import skimage.segmentation as sksegmentation
    # import skimage.measure as skmeasure 
    import numpy as np 
    
    
    uniq_regions = np.setdiff1d(np.unique(labels), 0)
    
    # get the border annotation. 
    border_px = sksegmentation.find_boundaries(labels,connectivity=2, mode='inner')
    
    border_tracks = border_px[tracks[0,:,0].astype(np.int32), 
                              tracks[0,:,1].astype(np.int32)].copy()
    lab_tracks = labels[tracks[0,:,0].astype(np.int32), 
                        tracks[0,:,1].astype(np.int32)].copy()
    
    
    # create a blank 
    blank_color = sksegmentation.mark_boundaries(np.ones(labels.shape[:2]+(3,)), 
                                                 labels, 
                                                 color=(0,0,0))
    ax.imshow(blank_color)
    
    for rr in uniq_regions:
        select = np.logical_and(lab_tracks==rr, border_tracks==True)
        
        tracks_select = tracks[:,select>0].copy()
        
        if len(tracks_select)>0:
            
            region_coords = np.argwhere(labels==rr)
            # need to do an angle and distance sort. 
            # get the median center.
            med = np.nanmedian(region_coords, axis=0)
        
            min_ix = np.argmin(np.linalg.norm(region_coords - med[None,:], axis=1))
            med = region_coords[min_ix,:].copy() # don't need this cos not the center!. 
            
            theta = np.arctan2(tracks_select[0,:,0]-med[0], tracks_select[0,:,1]-med[1])
            radius = np.linalg.norm(tracks_select[0,:] - med[None,:], axis=1)
            
            sort_ix = np.lexsort((radius, theta))
            
            N = tracks_select.shape[1]
            indices = np.arange(0,N, np.maximum(N//samples,2))
            
            tracks_select = tracks_select[:,sort_ix].copy() # apply the sort!. 

            ax.plot(tracks_select[:,indices,1], 
                    tracks_select[:,indices,0], 
                              color=color, lw=lw)
            ax.scatter(med[1], 
                       med[0], 
                        color='k', s=10, zorder=1000)
            
    return []
