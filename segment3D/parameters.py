import numpy as np 

def get_preprocess_params():
    
    # all display
    params = {# Image scaling
              'factor':1, # Isotropic Scaling Factor, positive-valued, no unit 
              'voxel_res':[1,1,1], # Pixel Size (to correct for anisotropic imaging), no unit
              # uneven illumination correction
              'do_bg_correction': True, # tickbox, name: Do Uneven Illumination Correction (if not checked don't do this)
              'bg_ds': 16, # only if do_bg_correction is True, name: Downsample Scale Factor, positive-value
              'bg_sigma' : 5, # only if do_bg_correction is True, name: Smooth Sigma, positive-value
              # image intensity normalization
              'normalize_min': 2., # Lower Intensity Cutoff; as percent, 0-100
              'normalize_max': 99.8, # Upper Intensity Cutoff; as percent, 0-100
              # for multi-channel 3D images 
              'do_avg_imgs': False, # name: average channels, make multichannel images into one channel. 
              'avg_func_imgs': np.nanmean} # name: average channel function, not used if do_avg_imgs is False, otherwise uses the function to do average of channels
    
    return params

# Step2 : run Cellpose 2D. 
def get_Cellpose_autotune_params():
    
    params = {'hist_norm': False, # checkbox, name: Histogram Normalize Image
              'cellpose_modelname': 'cyto', # name: Cellpose Model, select from={cyto, cyto2, cyto3, nuclei}
              'cellpose_channels' : 'grayscale', # name: Cellpose Color, select from={grayscale, 'color'(Note:R=nuclei, G=cytoplasm)} # grayscale as default. channels = [2,1] # else:, IF YOU HAVE G=cytoplasm and R=nucleus
              'ksize': 15, # int, name: Contrast Score Neighborhood Size, must be > 1. 
              'use_Cellpose_auto_diameter': False, # bool, name: Use Cellpose Default for Setting Diameter Parameter
              'gpu' : True, # HIDE on GUI. give warning. GPU is highly recommended for speed!  
              'best_diam':None, # name: Cellpose Diameter; accepts: positive integers >= 1. For GUI, can have automatic checked or user input. Explanation: Cellpose diameter (mean size of cells, px)  
              'model_invert' : False, # name: Invert Cellpose Model, 
              'test_slice' : None, # name: Representative 2D Slice #. integer (0-length of first dimension of image). For GUI, can have automatic or user input.
              'diam_range' : np.arange(10,121,2.5), # name: Diameter Scan Range, minimum diameter: 10, maximum_diameter: 120 (+ 1), increment: 2.5. these can all be floats. postive valued# this is the scan range for automatic diameter setting.  
              'smoothwinsize' : 5, # int, name: Smooth Contrast Function Window Size, >= 3 
              'histnorm_kernel_size':(64,64), # if hist_norm, name: Kernel Size of 2D Histogram Equalization, 2-tuple of ints. positive integers  
              'histnorm_clip_limit':0.05, # if hist_norm, name: Clip Limit in 2D Histogram Equalization, float [percentage of pixels to saturate]  
              'use_edge': True, # name: Use Edge Magnitude to Set Best Slice; if test_slice is not set, on GUI do dropdown: true-1. Use edge strength to determine optimal slice.,false-2. Use maximum intensity to determine optimal slice
              'show_img_bounds': (1024,1024), # HIDE ---- just visualization img[:1024,:1024] 
              'saveplotsfolder':None, # HIDE 
              'use_prob_weighted_score':True, # HIDE 
              'debug_viz': True} # HIDE -- name: Display Debugging Images for Automatic Diameter Inference. if True, saves image per diameter into outputfolder to check the inferred is correct.
            
    return params
    
# Step 3: MAIN ALGORITHM 
def get_2D_to_3D_aggregation_params():
    # Expose top-level as individual panels. Their names:
    # Combine 2D to 3D Gradients
    # Combine 2D to 3D Probabilities
    # Postprocess Combined 3D Probability
    # 3D Gradient Descent
    # 3D Cell Clustering # 'connected_component'
    # Distance Transform # 'indirect_method'
    params = {'combine_cell_gradients': {'ksize':1,  #int, name: Neighborhood Size, positive integer >=1 
                                         'alpha':0.5, # name : Pseudo Count Smoothing, positive float. 
                                         'eps':1e-20, # HIDE 
                                         'smooth_sigma':1, # name: Presmooth 2D Gradients (from 2D Segmentations before Combining), positive float >= 0
                                         'post_sigma':0}, # name: Smooth Combined 3D Gradients, positive float >= 0
              'combine_cell_probs': {'ksize':1,  #int, name: Neighborhood Size, positive integer >=1 
                                     'alpha':0.5, # name : Pseudo Count Smoothing, positive float. 
                                     'eps':1e-20, # HIDE 
                                     'cellpose_prob_mask':True, # -HIDE QZ only used when step 2 is cellpose; name: Normalize Cellpose Output Probabilities, Should be False ONLY if they are using cellpose for step 2 and have normalized the output themselves (note: probabilities should then be normalized to 0-1). 
                                     'smooth_sigma':0,  # name: Smooth Combined 3D Binary, positive float >= 0
                                     'threshold_level':1,  # name: Binary Threshold Level, #int 0<= threshold_level <= threshold_n_levels - 2; this should specify one of the possible threshold levels. it should not exceed the threshold_n_levels e.g. if threshold_n_levels=3, possibilities is 0 or 1 corresponding to the 1st/2nd levels. Generally n threshold levels means n-1 possible integers. 
                                     'threshold_n_levels':3,  #int 2<=threshold_n_levels<= inf # name: Number of Binary Threshold Partitions, this divides the binary into this many partitions. the number of threshold levels is 1 less than this number. 
                                     'apply_one_d_p_thresh':True, #bool # name: Enforce 1 Decimal Point for Cell Probability. This floors the inferred automatic threshold so that it is 1 decimal point. 
                                     'prob_thresh':None, # float 0 < prob_thresh < 1, name: Cell Probability Threshold, manual input to set the probability threshold, overriding the threshold_n_levels settings.
                                     'min_prob_thresh':0.0}, # float 0<= min_prob_thresh < 1, name: Minimum Cell Probability Threshold. This is the minimum possible cutoff for probability threshold, in case the automatic sets too low a number. 
               'postprocess_binary': {'binary_closing':1,  #int, name: Binary Closing Kernel Size. positive integer >=0 
                                       'remove_small_objects':1000,  #int, name: Small Objects Size Cutoff [voxels], positive integer. Removes all segmentations with volume less than specified number
                                       'binary_dilation': 1,  #int, name: Binary Dilation Kernel Size. positive integer >=0 
                                       'binary_fill_holes':False, # bool, name: Binary Infill Holes ?, 
                                       'extra_erode':0},  #int # name: Additional Binary Erosion Kernel Size, positive integer >= 0 
               'gradient_descent': {'gradient_decay':0.0, # name: Temporal Decay. positive float >=0, if greater, better for elongated. 
                                              'do_mp': False, # name: Do Multiprocess Version. 
                                              'tile_shape':(128,256,256), # name: Subvolume Shape #int >0 # only if do_mp is set. Subvolume size, integers that cannot be bigger than original image size
                                              'tile_aspect':(1,2,2), # name: Subvolume Aspect Ratio # float > 0 # only if do_mp is set. this specifies desired subvolume aspect ratios
                                              'tile_overlap_ratio':0.25, # name: Subvolume Overlap Fraction # fraction >=0 <1, sets the spatial overlap between subvolume times [only if do_mp True]
                                              'n_iter':200, # --HIDE QZ; name: Number of Iterations. postive integer (not usually changed) 
                                              'delta':1., # --HIDE ; name: Step Size of Gradient Descent [not usually changed]
                                              'momenta':0.98, # --HIDE QZ; name: Gradient Momentum, 0-1, [not usually changed] - display 0.95/0.98
                                              'eps':1e-12, # HIDE
                                              'use_connectivity': False, # not used 
                                              'connectivity_alpha':0.75, # not used 
                                              'interp':False, # not used 
                                              'binary_mask_gradient': False, # not used 
                                              'debug_viz': False, # if True, output images to output folder. # HIDE - QZ TODO, change default to true for viewer?
                                              'renorm_gradient': False, # HIDE 
                                              'sampling':1000, # --HIDE QZ; only if debug_viz=True; name: Sampling Spacing of 3D Points for Visualization # integer, decrease for more points in visualization, increase for less points. []
                                              'track_percent':0, # HIDE
                                              'rand_seed':0, # HIDE
                                              'ref_initial_color_img':None, # HIDE - QZ; only if debug_viz = True; name: Coloring for 3D Point of Points for Visualization. Must be (n_points,3) array. Default is just one color
                                              'ref_alpha':0.5, # name: Transparency of Visualized Points, >= 0 <=1
                                              'saveplotsfolder':None, # HIDE
                                              'viewinit':((0,0))}, # --HIDE QZ; only if debug_viz=True; name: View Angle of 3D Visualization. 2-tuple of floats, used only in the 3D debugging visualization. 
               'connected_component': {'min_area': 5,  #int, # name: Smallest Cluster Size Cutoff,  
                                      'smooth_sigma':1., # name: Smoothing Sigma for Computing Point Density, positive float>=0. 
                                      'thresh_factor': 0}, # QZ change default from None to 0 #float, instead of None, can use 0. # name: Thresholding Adjustment Factor (threshold = mean + thresh_factor * std); Multiplier for Adjusting Threshold of Point Density = mean + thresh_factor * std. (default: None = 0)
              'indirect_method': {'dtform_method': 'cellpose_improve', #name: Distance Transform Method, #dropdown, select from {'cellpose_improve' (note: Heat Diffusion), 'edt' (Note: Euclidean Distance Transform), 'fmm' (Note: Geodesic Centroid), 'fmm_skel' (Note: Geodesic Skeleton), 'cellpose_skel' (Note: Diffusion Skeleton)
                                  'iter_factor': 5,  #int, # HIDE, not used 
                                  'power_dist': None, # float, >0 <=1, name: Distance Transform Exponent # Exponent to Raise Distance Transform to Boost Gradients (only applied if using diffusion transforms i.e. 'cellpose_improve'), 
                                  'n_cpu': None, #int >0 # name: Number of CPU to Use (default is empty - use max in sys).
                                  'edt_fixed_point_percentile':0.01, # float, >=0 <=1 # name: Euclidean Distance Transform Threshold Percentile # Percentile to Threshold Euclidean Distance Transform to Constrain the Centroid Placement. Only used if distance tranform is 'fmm' or 'cellpose_improve'
                                  'smooth_binary': 1,  # positive float >= 1, # name: Smooth Combined 3D Binary # Sigma for Smoothing Combined Binaries Derived from Segmentation Labels.
                                  'smooth_skel_sigma':3}} # positive float >= 1, # name: Smooth Skeleton # Smoothing Sigma to Get a Better Medial-axis Skeleton Devoid of Many Small Branches.  Only used for 'cellpose_skel' / 'fmm_skel' distance transforms
    
    return params


def get_postprocess_segmentation_params():
    
    params = {'size_filters': {'min_size': 200, # int >0, name: The Smallest Allowable Cell Size [voxels] 
                               'max_size': None, # int >0, name: The Largest Allowable Cell Size [voxels]
                               'max_size_factor': 10, # float >0, name: Multiplier for Setting Maximum Cell Size (= mean + max_size_factor * std) 
                               'do_stats_filter' : True}, # Name: Perform Maximum Size Filtering # bool, if set, implements the above, max_size_factor filtering for maximum object size. 
              'flow_consistency' : {'flow_threshold': 0.85, # float >0, # name: Maximum Mean Squared Error Between Reconstructed 3D Gradients and Combined 2D-to-3D Gradients.
                                    'do_flow_remove': True, # name: Perform Flow Consistency Filter # bool, if True, implements the flow threshold
                                    'edt_fixed_point_percentile':0.01, # same as above;;; float >=0 <=1; name: Percentile to Threshold Euclidean Distance Transform to Constrain the Centroid Placement. Note: Should be same as that used in the direct method if this was the way 3D segmentations were generated.
                                    'dtform_method':'cellpose_improve', # same as above;;; #name: Distance Transform Method, #dropdown, select from {'cellpose_improve' (note: heat diffusion) step 2 cellpose seg choose this only, 'edt' (Note: Euclidean distance transform), 'fmm' (Note: Geodesic Centroid), 'fmm_skel' (Note: Geodesic Skeleton), 'cellpose_skel' (Note: Diffusion Skeleton)
                                    'iter_factor': 5,  # HIDE. 
                                    'power_dist': None, # same as above;;; # float, >0 <=1, name: Exponent to Raise Distance Transform to Boost Gradients (only applied if using diffusion transforms i.e. 'cellpose_improve'), 
                                    'smooth_skel_sigma':3, # same as above;;; # positive float >= 1, # name: Smoothing Sigma to Get a Better Medial-Axis Skeleton Devoid of Many Small Branches. Only used for cellpose_skel / fmm_skel distance transforms
                                    'n_cpu': None} }  #int >0 # name: Number of CPU to Use (default is empty - use max in sys). 
              
    return params
    

def get_label_diffusion_params():
    # Expose top-level as individual panels. Their names:
    # Diffusion
    # Guide Image
    params = {'diffusion': {'n_cpu':None,  #int >0 # name: Number of CPU to Use (default is empty - use max in sys). 
                            'refine_clamp': 0.7, # float >=0 <=1, # name: Clamping Ratio (Note; 0.0: no clamping and enforcement of input segmentation, 1.0: fully enforce input segmentation )
                            'refine_iters': 50,  #int >0, # name: Number of Iterations of Diffusion. 
                            'refine_alpha': 0.5, #float, >=0 <=1. # name: Weighting of Affinity Graph Between 0 and 1 (0: fully based on spatial separation of voxels, 1: fully based on guide image intensity gradients)
                            'pad_size': 25,  #int >=0 # name: Number of Pixels to Pad (Around Individual Cell Crops.)
                            'multilabel_refine':False, # bool, # name: Perform Refinement Jointly Considering All Cells in the Individual Cell Crops.
                            'noprogress_bool':True, # bool, # name : Suppress Progressbar (recommended to be True, if lots of cells)
                            'affinity_type':'heat'}, # HIDE, we always just use 'heat'  
              # to ensure guide img input is scaled to have intensity 0-1. 
              'guide_img': {'pmin': 0, # Lower Intensity Cutoff as Percent, float >=0 <=100 but < pmax
                            'pmax':100}} # Upper Intensity Cutoff as Percent, float  >=0 <=100, but > pmin

    return params


def get_guided_filter_params():
    # Expose top-level as individual panels. Their names:
    # Guide Filter (subpanel Collision with 4 collision_ parmas, subpanel Guide Image with last 2 params)
    # Ridge Filter 
    params = {'guide_filter': {'radius': 25,  #int >0, name: Radius
                               'eps': 1e-4, # float >0, name: Regularization Strength, the smaller the value the stronger the filter.
                               'n_cpu': None,  #int >0 , # name: Number of CPU to Use (default is empty - use max in sys). 
                               'pad_size': 25,  #int >=0 , # name: Number of Pixels to Pad (Around Individual Cell Crops.)
                               'size_factor': 0.75, # float > 0, # name: Proportion of Cell's Mean Bounding Box Diameter (to Consider for Recovering Protrusions). can be greater than 1.  
                               'min_protrusion_size': 15., # float >= 0, # name: Minimum Protrusion Size [voxel]; Minimum Voxel Size Around Segmentation to Consider for Recovering Protrusions. 
                               'adaptive_radius_bool': False, # bool, # name: Use Automatic Radius ; Set Radius for Recovering Protrusions Proportionate to Each Cell's Mean Bounding Box Width. 
                               'mode': 'normal', # dropdown/list either of 'normal' or 'additive'. # name: Operating Mode for Recovering Protrusions; Adding Protrusions to Input Segmentation. If 'normal' the refined replaces the old. If 'additive' the input segmentation is slightly eroded and binary combined with the refined. Additive is useful when marker is membrane and therefore cell internal is dark. 
                               'threshold_level':0, # change default from -1 to 0 to make more sense on GUI-QZ;;#int 0<= threshold_level <= threshold_n_levels - 2;  # name: Binary Threshold Level; Binary Threshold Level for Rebinarizing Guided Filter Output, this should specify one of the possible threshold levels. it should not exceed the threshold_n_levels e.g. if threshold_n_levels=3, possibilities is 0 or 1 corresponding to the 1st/2nd levels. Generally n threshold levels means n-1 possible integers. 
                               'threshold_nlevels':2, #int 2<=threshold_n_levels<= inf # name: Number of Binary Threshold Partitions; (Number of Otsu Thresholding Partitions). Note: this divides the guided filter output into this many partitions. the number of threshold levels is 1 less than this number. 
                               'use_int':False, # bool, name: Use Individual Cell Segmentations as Integer Type (if True) or as Floating (if False). Note: guided filtering produces different outputs. Integer may help better enhance image edges but need to run this part in 'additive' mode, whereas float retains complete segmentation of cell core and can use 'normal' mode.  
                               'collision_erode':2,  #int >=0, name: Protrusion Erosion Kernel Size (to Avoid Merging with Other Cells)  # this is to prevent overlapping into another cell's area. 
                               'collision_close':3,  #int >=0, name: Protrusion Closing Kernel Size (to Fill in Holes After Rebinarizing Guided Filtered Segmentation), # this is to close after resolving overlapping into another cell's area. 
                               'collision_dilate':0,  #int >=0, name: Protrusion Dilation Kernel Size (to Recover from Initial Erosion to Separate from Neighboring Cells).  
                               'collision_fill_holes':True, # name: Protrusion Fill Holes
                               # only if mode is 'additive'. Note: only one of the base_dilate or base_erode should apply. Set to 0 to deactivate the other. 
                               'base_dilate':0,  #int >=0, name: Cell Dilation Kernel Size (of Input Cell Segmentation), prior to combining binary of guided filter outputs
                               'base_erode':5},  #int >=0, name: Cell Erosion Kernel Size (of Input Cell Segmentation), prior to combining binary of guided filter outputs. 
              'ridge_filter' : {'sigmas': [3],  #list of floats, at least one element, >0. name: List of Sizes [px] for Multiscale Ridge Filtering (List of Sizes [px] to Consider for Multiscale Ridge Feature Image Filtering)
                                'black_ridges': False, # bool. name: Ridges Dark; Ridges are Dark (Set True) else, Ridges are Bright (set False, default, as cell borders are bright)
                                'mix_ratio': 0.5, # float, >=0 <=1. name: Weighting of Ridge Filtered Guided Image [0: only input guided image, 1: only ridge enhanced image]. 
                                'do_ridge_enhance':False, # bool, name: Use Ridge Filter; Compute Ridge Enhanced Guided Image?, note: the ridge enhanced can then be combined with input guide image to create the final one for refinement. 
                                'do_multiprocess_2D':True, # bool, name: Computer Using 2D Multiprocessing (to Perform the 3D Ridge Enhancement). Recommended for large image volumes
                                'low_contrast_fraction':0.05, # float >=0 <=1. name: Low-Contrast Image Cutoff ; Percentile Cutoff for Designating 2D Image Slice as Low-Contrast. Note: low-contrast 2D slices are not ridge-enhanced. This only applies if ridge-enhancement and multiprocess option is set.
                                'n_cpu': None,  #int >0, name:  Number of CPU to Use (default is empty - use max in sys). 
                                # to ensure ridge enhanced img is scaled to have intensity 0-1. 
                                'pmin':2, # same as above; float[0-100] # name: Lower Intensity Cutoff of Ridge-Enhanced Image as Percent, 0-100
                                'pmax':99.8}, # same as above; float[0-100] # name: Upper Intensity Cutoff of Ridge-Enhanced Image as Percent, 0-100
              # to ensure guide img input is scaled to have intensity 0-1. 
              'guide_img': {'pmin': 0, # same as above; float[0-100] # name: Lower Intensity Cutoff as Percent, 0-100
                            'pmax':100}}# same as above; float[0-100] # name: Upper Intensity Cutoff as Percent, 0-100

    return params
    
    
    
    
    
    
