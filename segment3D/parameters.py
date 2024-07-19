import numpy as np 

def get_preprocess_params():
    
    # all display
    params = {# Image scaling
              'factor':1, # isotropic scaling factor, positive-valued, no unit 
              'voxel_res':[1,1,1], # pixel size (to correct for anisotropic imaging), no unit
              # uneven illumination correction
              'do_bg_correction': True, # tickbox, name: do uneven illumination correction (if not checked don't do this)
              'bg_ds': 16, # only if do_bg_correction is True, name: downsample scale factor, positive-value
              'bg_sigma' : 5, # only if do_bg_correction is True, name: smooth_sigma, positive-value
              # image intensity normalization
              'normalize_min': 2., # lower intensity cutoff as percent, 0-100
              'normalize_max': 99.8, # upper intensity cutoff as percent, 0-100
              # for multi-channel 3D images 
              'do_avg_imgs': False, # name: average channels, make multichannel images into one channel. 
              'avg_func_imgs': np.nanmean} # name: average channel function, not used if do_avg_imgs is False, otherwise uses the function to do average of channels
    
    return params

# Step2 : run Cellpose 2D. 
def get_Cellpose_autotune_params():
    
    params = {'hist_norm': False, # checkbox, name: histogram normalize image
              'cellpose_modelname': 'cyto', # name: cellpose model, select from={cyto, cyto2, cyto3, nuclei}
              'cellpose_channels' : 'grayscale', # name: cellpose color, select from={grayscale, 'color'(Note:R=nuclei, G=cytoplasm)} # grayscale as default. channels = [2,1] # else:, IF YOU HAVE G=cytoplasm and R=nucleus
              'ksize': 15, # int, name: contrast score neighborhood size, must be > 1. 
              'use_Cellpose_auto_diameter': False, # bool, name: Use Cellpose default for setting diameter parameter 
              'gpu' : True, # give warning. GPU is highly recommended for speed!  
              'best_diam':None, # name: cellpose diameter; accepts: positive integers >= 1. For GUI, can have automatic checked or user input. Explanation: Cellpose diameter (mean size of cells, px)  
              'model_invert' : False, # name: invert cellpose model, 
              'test_slice' : None, # name: representative 2D slice #. integer (0-length of first dimension of image). For GUI, can have automatic or user input.
              'diam_range' : np.arange(10,121,2.5), # name: diameter scan range, minimum diameter: 10, maximum_diameter: 120 (+ 1), increment: 2.5. these can all be floats. postive valued# this is the scan range for automatic diameter setting.  
              'smoothwinsize' : 5, # int, name: smooth contrast function window size, >= 3 
              'histnorm_kernel_size':(64,64), # if hist_norm, name: kernel size of 2D histogram equalization, 2-tuple of ints. positive integers  
              'histnorm_clip_limit':0.05, # if hist_norm, name: clip limit in 2D histogram equalization, float [percentage of pixels to saturate]  
              'use_edge': True, # if test_slice is not set, dropdown: 1. use edge strength to determine optimal slice., 2. use maximum intensity to determine optimal slice
              'show_img_bounds': (1024,1024), # hide ---- just visualization img[:1024,:1024] 
              'saveplotsfolder':None, # hide 
              'use_prob_weighted_score':True, # hide 
              'debug_viz': True} # name: Display debugging images for automatic diameter inference. if True, saves image per diameter into outputfolder to check the inferred is correct.
            
    return params
    
# Step 3: MAIN ALGORITHM 
def get_2D_to_3D_aggregation_params():
    
    # Expose top-level as individual panels?
    params = {'combine_cell_gradients': {'ksize':1,  #int, name: neighborhood size, positive integer >=1 
                                         'alpha':0.5, # name : pseudo count smoothing, positive float. 
                                         'eps':1e-20, # hide 
                                         'smooth_sigma':1, # name: presmooth gradients from 2D segmentations before combining, positive float >= 0
                                         'post_sigma':0}, # name: smooth combined 3D gradients, positive float >= 0
              'combine_cell_probs': {'ksize':1,  #int, name: neighborhood size, positive integer >=1 
                                     'alpha':0.5, # name : pseudo count smoothing, positive float. 
                                     'eps':1e-20, # hide 
                                     'cellpose_prob_mask':True, # name: normalize cellpose output probabilities, Should be False if not using Cellpose models with a note that probabilities should then be normalized to 0-1. 
                                     'smooth_sigma':0,  # name: smooth combined 3D binary, positive float >= 0
                                     'threshold_level':-1,  # name: binary threshold level, #int, this should specify one of the possible threshold levels. it should not exceed the threshold_n_levels e.g. if threshold_n_levels=3, possibilities is 0 or 1 corresponding to the 1st/2nd levels. Generally n threshold levels means n-1 possible integers. 
                                     'threshold_n_levels':3,  #int # name: number of binary threshold partitions, this divides the binary into this many partitions. the number of threshold levels is 1 less than this number. 
                                     'apply_one_d_p_thresh':True, #bool # name: enforce 1 decimal point for cell probability. This floors the inferred automatic threshold so that it is 1 decimal point. 
                                     'prob_thresh':None, # float 0-1, name: cell probability threshold, manual input to set the probability threshold, overriding the threshold_n_levels settings.
                                     'min_prob_thresh':0.0}, # float 0-1, name: minimum cell probability threshold. This is the minimum possible cutoff for probability threshold, in case the automatic sets too low a number. 
               'postprocess_binary': {'binary_closing':1,  #int, name: binary closing kernel size. positive integer >=0 
                                       'remove_small_objects':1000,  #int, name: small objects size cutoff [voxels], positive integer. Removes all segmentations with volume less than specified number
                                       'binary_dilation': 1,  #int, name: binary dilation kernel size. positive integer >=0 
                                       'binary_fill_holes':False, # bool, name: binary infill holes ?, 
                                       'extra_erode':0},  #int # name: additional binary erosion kernel size, positive integer >= 0 
               'gradient_descent': {'gradient_decay':0.0, # name: temporal decay. positive float, if greater, better for elongated. 
                                              'do_mp': False, # name: do multiprocess version. 
                                              'tile_shape':(128,256,256), # name: subvolume shape #int # only if do_mp is set. Subvolume size, integers that cannot be bigger than original image size
                                              'tile_aspect':(1,2,2), # name: subvolume aspect ratio # only if do_mp is set. this specifies desired subvolume aspect ratios
                                              'tile_overlap_ratio':0.25, # name: subvolume overlap fraction # fraction 0-1, sets the spatial overlap between subvolume times [only if do_mp True]
                                              'n_iter':200, # name: number of iterations. postive integer (not usually changed) 
                                              'delta':1., # name: step size of gradient descent [not usually changed] - hide
                                              'momenta':0.98, # name: gradient momentum, 0-1, [not usually changed] - display 0.95/0.98
                                              'eps':1e-12, # hide
                                              'use_connectivity': False, # not used 
                                              'connectivity_alpha':0.75, # not used 
                                              'interp':False, # not used 
                                              'binary_mask_gradient': False, # not used 
                                              'debug_viz': False, # if True, output images to output folder. 
                                              'renorm_gradient': False, # hide 
                                              'sampling':1000, # name: Sampling spacing of 3D points for visualization # only if debug_viz=True # integer, decrease for more points in visualization, increase for less points. []
                                              'track_percent':0, # hide
                                              'rand_seed':0, # hide
                                              'ref_initial_color_img':None, # name: Coloring for 3D point of points for visualization # only if debug_viz = True. Must be (n_points,3) array. Default is just one color
                                              'ref_alpha':0.5, # name: transparency of visualized points, 0-1
                                              'saveplotsfolder':None, # hide
                                              'viewinit':((0,0))}, # name: view angle of 3D visualization. 2-tuple of floats, used only in the 3D debugging visualization. 
               'connected_component': {'min_area': 5,  #int, # name: smallest cluster size cutoff,  
                                      'smooth_sigma':1., # name : smoothing sigma for computing point density, positive float>=0. 
                                      'thresh_factor': None}, #float, instead of None, can use 0. # name: multiplier for adjusting threshold of point density = mean + thresh_factor * std. (default: None = 0)
              'indirect_method': {'dtform_method': 'cellpose_improve', #name: distance transform method, #dropdown, select from {'cellpose_improve' (note: heat diffusion), 'edt' (Note: Euclidean distance transform), 'fmm' (Note: Geodesic Centroid), 'fmm_skel' (Note: Geodesic Skeleton), 'cellpose_skel' (Note: Diffusion Skeleton)
                                  'iter_factor': 5,  #int, # hide, not used 
                                  'power_dist': None, # float, 0-1, name: exponent to raise distance transform to boost gradients (only applied if using diffusion transforms), 
                                  'n_cpu': None,  #int # name: number of CPU to use. 
                                  'edt_fixed_point_percentile':0.01, # float, # name: percentile to threshold euclidean distance transform to constrain the centroid placement. Only used if distance tranform is 'fmm' or 'cellpose_improve'
                                  'smooth_binary': 1,  # positive float >= 1, # name: sigma for smoothing combined binaries derived from segmentation labels.
                                  'smooth_skel_sigma':3}} # positive float >= 1, # name: smoothing sigma to get a better medial-axis skeleton devoid of many small branches.  Only used for cellpose_skel / fmm_skel distance transforms
    
    return params


def get_postprocess_segmentation_params():
    
    params = {'size_filters': {'min_size': 200, # int, name: The smallest allowable cell size [voxels] 
                               'max_size_factor': 10, # float, name: multiplier for setting maximum cell size = mean + max_size_factor * std 
                               'do_stats_filter' : True}, # name: perform maximum size filtering # bool, if set, implements the above, max_size_factor filtering for maximum object size. 
              'flow_consistency' : {'flow_threshold': 0.85, # float >0, # name: maximum mean squared error between reconstructed 3D gradients and combined 2D-to-3D gradients.
                                    'do_flow_remove': True, # name: perform flow consistency filter # bool, if True, implements the flow threshold
                                    'edt_fixed_point_percentile':0.01, # name: # name: percentile to threshold euclidean distance transform to constrain the centroid placement. Note: Should be same as that used in the direct method if this was the way 3D segmentations were generated.
                                    'dtform_method':'cellpose_improve', #name: distance transform method, #dropdown, select from {'cellpose_improve' (note: heat diffusion), 'edt' (Note: Euclidean distance transform), 'fmm' (Note: Geodesic Centroid), 'fmm_skel' (Note: Geodesic Skeleton), 'cellpose_skel' (Note: Diffusion Skeleton)
                                    'iter_factor': 5,  # hide. 
                                    'power_dist': None, # float, 0-1, name: exponent to raise distance transform to boost gradients (only applied if using diffusion transforms), 
                                    'smooth_skel_sigma':3, # positive float >= 1, # name: smoothing sigma to get a better medial-axis skeleton devoid of many small branches. Only used for cellpose_skel / fmm_skel distance transforms
                                    'n_cpu': None} }  #int # name: number of CPU to use. 
              
    return params
    

def get_label_diffusion_params():
    
    params = {'diffusion': {'n_cpu':None,  #int # name: number of CPU to use. 
                            'refine_clamp': 0.7, # float, # name: clamping ratio (Note; 0.0: no clamping and enforcement of input segmentation, 1.0: fully enforce input segmentation )
                            'refine_iters': 50,  #int, # name: number of iterations of diffusion. 
                            'refine_alpha': 0.5, #float, 0-1. # name: weighting of affinity graph between 0: fully based on spatial separation of voxels, or 1: fully based on guide image intensity gradients
                            'pad_size': 25,  #int # name: number of pixels to pad around individual cell crops.
                            'multilabel_refine':False, # bool, # name: perform refinement jointly considering all cells in the individual cell crops.
                            'noprogress_bool':True, # bool, # name : suppress progressbar (recommended to be True, if lots of cells)
                            'affinity_type':'heat'}, # hide, we always just use 'heat'  
              # to ensure guide img input is scaled to have intensity 0-1. 
              'guide_img': {'pmin': 0, # lower intensity cutoff as percent, 0-100
                            'pmax':100}} # upper intensity cutoff as percent, 0-100

    return params


def get_guided_filter_params():
    
    params = {'guide_filter': {'radius': 25,  #int
                               'eps': 1e-4,
                               'n_cpu': None,  #int, # name: number of CPU to use. 
                               'pad_size': 25,  #int, # name: number of pixels to pad around individual cell crops.
                               'size_factor': 0.75, # float> 0, # name: proportion of cell's mean bounding box diameter to consider for recovering protrusions. can be greater than 1.  
                               'min_protrusion_size': 15., # float >= 0, # name: minimum voxel size around segmentation to consider for recovering protrusions. 
                               'adaptive_radius_bool': False, # bool, # name: set radius for recovering protrusions proportionate to each cell's mean bounding box width. 
                               'mode': 'normal', # dropdown/list either of 'normal' or 'additive'. # name: operating mode for adding  protrusions to input segmentation. If 'normal' the refined replaces the old. If 'additive' the input segmentation is slightly eroded and binary combined with the refined. Additive is useful when marker is membrane and therefore cell internal is dark. 
                               'threshold_level':-1,  #int, # name: binary threshold level for rebinarizing guided filter output, this should specify one of the possible threshold levels. it should not exceed the threshold_n_levels e.g. if threshold_n_levels=3, possibilities is 0 or 1 corresponding to the 1st/2nd levels. Generally n threshold levels means n-1 possible integers. 
                               'threshold_nlevels':2, #int # name: number of Otsu thresholding partitions. Note: this divides the guided filter output into this many partitions. the number of threshold levels is 1 less than this number. 
                               'use_int':False, # bool, name: use individual cell segmentations as integer type (if True) or as floating (if False). Note: guided filtering produces different outputs. Integer may help better enhance image edges but need to run this part in 'additive' mode, whereas float retains complete segmentation of cell core and can use 'normal' mode.  
                               'collision_erode':2,  #int, name: erosion kernel size to avoid merging with other cells  # this is to prevent overlapping into another cell's area. 
                               'collision_close':3,  #int, name: closing kernel size to fill in holes after rebinarizing guided filtered segmentation, # this is to close after resolving overlapping into another cell's area. 
                               'collision_dilate':0,  #int, name: dilation kernel size to recover from initial erosion to separate from neighboring cells.  
                               'collision_fill_holes':True,
                               # only if mode is 'additive'. Note: only one of the base_dilate or base_erode should apply. Set to 0 to deactivate the other. 
                               'base_dilate':0,  #int, name: dilation kernel size of input cell segmentation, prior to combining binary of guided filter outputs
                               'base_erode':5},  #int, name: erosion kernel size of input cell segmentation, prior to combining binary of guided filter outputs. 
              'ridge_filter' : {'sigmas': [3],  #list of integers. name: list of sizes [px] to consider for multiscale ridge feature image filtering 
                                'black_ridges': False, # bool. name: are ridges dark (Set True) else, ridges are bright (set False, default, as cell borders are bright)
                                'mix_ratio': 0.5, # float, 0-1. name: weighting of ridge filtered guided image [0: only input guided image, 1: only ridge enhanced image]. 
                                'do_ridge_enhance':False, # bool, name: compute ridge enhanced guided image?, note: the ridge enhanced can then be combined with input guide image to create the final one for refinement. 
                                'do_multiprocess_2D':True, # bool, name: use 2D multiprocessing to perform the 3D ridge enhancement. Recommended for large image volumes
                                'low_contrast_fraction':0.05, # float[0-1]. name: percentile cutoff for designating 2D image slice as low-contrast.Note: low-contrast 2D slices are not ridge-enhanced. This only applies if ridge-enhancement and multiprocess option is set.
                                'n_cpu': None,  #int, name:  number of CPU to use. 
                                # to ensure ridge enhanced img is scaled to have intensity 0-1. 
                                'pmin':2, # float[0-100] # name: lower intensity cutoff of ridge-enhanced image as percent, 0-100
                                'pmax':99.8}, # float[0-100] # name: upper intensity cutoff of ridge-enhanced image as percent, 0-100
              # to ensure guide img input is scaled to have intensity 0-1. 
              'guide_img': {'pmin': 0, # float[0-100] # name: lower intensity cutoff as percent, 0-100
                            'pmax':100}}# float[0-100] # name: upper intensity cutoff as percent, 0-100

    return params
    
    
    
    
    
    
