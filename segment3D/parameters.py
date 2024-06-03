import numpy as np 

def get_preprocess_params():
    
    params = {'factor':1,
              'voxel_res':[1,1,1], 
              'do_bg_correction': True, 
              'bg_ds': 16,
              'bg_sigma' : 5, 
              'normalize_min': 2.,
              'normalize_max': 99.8,
              'do_avg_imgs': False, 
              'avg_func_imgs': np.nanmean}
    
    return params

def get_Cellpose_autotune_params():
    
    params = {'hist_norm': False, 
              'cellpose_modelname': 'cyto',
              'cellpose_channels' : 'grayscale', # grayscale as default. channels = [2,1] # else:, IF YOU HAVE G=cytoplasm and R=nucleus
              'ksize': 15,
              'use_Cellpose_auto_diameter': False,
              'gpu' : True,
              'best_diam':None,
              'model_invert' : False,
              'test_slice' : None, 
              'hist_norm' : False, # was True
              'diam_range' : np.arange(10,121,2.5), 
              'smoothwinsize' : 5,
              'histnorm_kernel_size':(64,64), 
              'histnorm_clip_limit':0.05,
              'use_edge': True,
              'show_img_bounds': (1024,1024),
              'saveplotsfolder':None,
              'use_prob_weighted_score':True,
              'debug_viz': True}
            
    return params
    

def get_2D_to_3D_aggregation_params():
    
    params = {'gradient_descent': {'gradient_decay':0.0,
                                   'do_mp': False, # multiprocess is not the default
                                   'tile_shape':(128,256,256),
                                   'tile_aspect':(1,2,2),
                                   'tile_overlap_ratio':0.25,
                                   'n_iter':200, 
                                   'delta':1.,
                                   'momenta':0.98, 
                                   'eps':1e-12,
                                   'use_connectivity': False,
                                   'connectivity_alpha':0.75,
                                   'interp':False,
                                   'binary_mask_gradient': False,
                                   'debug_viz': False,
                                   'renorm_gradient': False, 
                                   'sampling':1000,
                                   'track_percent':0, 
                                   'rand_seed':0, 
                                   'ref_initial_color_img':None,
                                   'ref_alpha':0.5, 
                                   'saveplotsfolder':None,
                                   'viewinit':((0,0))},
              'combine_cell_probs': {'ksize':1, 
                                     'alpha':0.5, 
                                     'eps':1e-20, 
                                     'cellpose_prob_mask':True, 
                                     'smooth_sigma':0, 
                                     'threshold_level':-1, 
                                     'threshold_n_levels':3,
                                     'apply_one_d_p_thresh':True,
                                     'prob_thresh':None,
                                     'min_prob_thresh':0.0}, 
               'postprocess_binary': {'binary_closing':1, 
                                       'remove_small_objects':1000, 
                                       'binary_dilation': 1,
                                       'binary_fill_holes':False,
                                       'extra_erode':0},
              'combine_cell_gradients': {'ksize':1, 
                                         'alpha':0.5, 
                                         'eps':1e-20, 
                                         'smooth_sigma':1, 
                                         'post_sigma':0}, 
              'connected_component': {'min_area': 5,
                                      'smooth_sigma':1.,
                                      'thresh_factor': None},
              'indirect_method': {'dtform_method': 'cellpose_improve',
                                  'iter_factor': 5, 
                                  'power_dist': None,
                                  'n_cpu': None,
                                  'edt_fixed_point_percentile':0.01,
                                  'smooth_binary': 1,
                                  'smooth_skel_sigma':3}}
    
    return params


def get_postprocess_segmentation_params():
    
    params = {'size_filters': {'min_size': 200,
                               'max_size_factor': 10,
                               'do_stats_filter' : True},
              'flow_consistency' : {'flow_threshold': 0.85,
                                    'do_flow_remove': True,
                                    'edt_fixed_point_percentile':0.01,
                                    'dtform_method':'cellpose_improve',
                                    'iter_factor': 5, 
                                    'power_dist': None,
                                    'smooth_skel_sigma':3, 
                                    'n_cpu': None} }
              
    return params
    

def get_label_diffusion_params():
    
    params = {'diffusion': {'n_cpu':None,
                            'refine_clamp': 0.7, # increase to have less effect. 
                            'refine_iters': 50, 
                            'refine_alpha': 0.5,
                            'pad_size': 25,
                            'multilabel_refine':False, # if True, then will consider all segmented touching cells in refinement
                            'noprogress_bool':True,
                            'affinity_type':'heat'}, # <0.5 more diffusion >0.5 more image based. 
              'guide_img': {'pmin': 0,
                            'pmax':100}} 

    return params


def get_guided_filter_params():
    
    params = {'guide_filter': {'radius': 25,
                               'eps': 1e-4,
                               'n_cpu': None,
                               'pad_size': 25,
                               'size_factor': 0.75,
                               'min_protrusion_size': 15., 
                               'adaptive_radius_bool': False,
                               'mode': 'normal', 
                               'base_dilate':0,
                               'base_erode':5, 
                               'collision_erode':2, # this is to prevent overlapping into another cell's area. 
                               'collision_close':3, # this is to close after resolving overlapping into another cell's area. 
                               'collision_dilate':0,
                               'collision_fill_holes':True,
                               'threshold_level':-1, 
                               'threshold_nlevels':2,# 2 is normal otsu. 
                               'use_int':False}, 
              'ridge_filter' : {'sigmas': [3],
                                'black_ridges': False,
                                'mix_ratio': 0.5,
                                'do_ridge_enhance':False,
                                'do_multiprocess_2D':True,
                                'low_contrast_fraction':0.05,
                                'n_cpu': None, 
                                'pmin':2,
                                'pmax':99.8},
              'guide_img': {'pmin': 0,
                            'pmax':100}}

    return params
    
    
    
    
    
    
