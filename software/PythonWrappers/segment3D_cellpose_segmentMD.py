"""
This is the key function + script, used in the wrapper fcn (cellposeSegmentWrap) 
of one of the option ,CellposeSegmentationProcess.m, in step 2 in u-segment3D MATLAB package.

This script is used to integrate python package/script with MovieData matlab structure.

I wrapped this script as a Python function, and defined two input arguments, input_path and output_path, with return [].
Then call this segment3D_cellpose_segmentMD at bottom, since pyrunfile only runs script.
I added an input params = p, which was a structure in Matlab.

Example of how to use this test script in Matlab:
    inPath = '/path/to/folder/of/input/images'
    outPath = '/path/to/folder/for/saving/output' # will be created if not exist.
    pyrunfile("segment3D_cellpose_segmentMD.py", input_path = inPath, output_path = outPath, params = p)


It is based on python script:
/applications/PythonPackIntoMovieData/Segment3D/tutorials/single_cells/segment3D_cellpose_segment.py

Alse see the full Python script: segment_blebs_3D_qz.py.
Also see script by Felix Zhou in 
/python-applications/Segment3D/tutorials/single_cells/segment_blebs_3D.py

Qiongjing (Jenny) Zou, June 2024

"""
  
"""
Step 2: Cellpose Segmentation

"""
# only load the modules needed for this step - QZ:
    
import skimage.io as skio 
# import pylab as plt 
import numpy as np 
# import scipy.ndimage as ndimage
# import skimage.segmentation as sksegmentation 
import os 

import segment3D.parameters as uSegment3D_params # this is useful to call default parameters, and keep track of parameter changes and for saving parameters.  
# import segment3D.gpu as uSegment3D_gpu
import segment3D.usegment3d as uSegment3D
# import segment3D.filters as uSegment3D_filters 
# import segment3D.file_io as uSegment3D_fio

def segment3D_cellpose_segmentMD(input_path, output_path, params):
    #########################################
    # Set Preconditions
    #########################################

    inputFile = input_path

    saveFolderStep2 = output_path
    if not os.path.exists(saveFolderStep2):
            os.makedirs(saveFolderStep2)


    print("Starting u-segment3D Package Step 2 Cellpose Segmentation")

    ##############################
    ##############################
    ##############################

    # =============================================================================
    #     Read the image from previous step - QZ
    # =============================================================================

    img_preprocess = skio.imread(os.path.join(inputFile, 'preprocessed.tif'))

    # =============================================================================
    #     2. Run Cellpose 2D in xy, xz, yz with auto-tuning diameter to get cell probability and gradients, in all 3 views. 
    # =============================================================================
    # we are going to use our automated tuner to run cellpose scanning the stack in xy, xz, and yz to set an automatic diameter for each view
    # this will generate the predicted cell probabilites and gradients

    # get parameters from movieData: - QZ
    cellpose_segment_params = params

    # Convert string parameter from Matlab to a numpy function in Python:
    cellpose_segment_params['diam_range'] = eval(cellpose_segment_params['diam_range'])

    print('========== Cellpose segmentation parameters ========')
    print(cellpose_segment_params)    
    print('============================================')


    # this expects a multichannel input image and in the format (M,N,L,channels) i.e. channels last.
    img_preprocess = img_preprocess[...,None] # we generate an artificial channel

    # =============================================================================
    #     Run and Save Output:
    # =============================================================================

    # QZ: changed basename and savefolder from None, so the output can be saved.

    basename_xy = 'xy'
    basename_xz = 'xz'
    basename_yz = 'yz'


    #### 1. running for xy orientation. If the savefolder and basename are specified, the output will be saved as .pkl and .mat files 
    img_segment_2D_xy_diams, img_segment_3D_xy_probs, img_segment_2D_xy_flows, img_segment_2D_xy_styles = uSegment3D.Cellpose2D_model_auto(img_preprocess, 
                                                                                                                                           view='xy', 
                                                                                                                                           params=cellpose_segment_params, 
                                                                                                                                           basename=basename_xy, savefolder=saveFolderStep2)

    #### 2. running for xz orientation 
    img_segment_2D_xz_diams, img_segment_3D_xz_probs, img_segment_2D_xz_flows, img_segment_2D_xz_styles = uSegment3D.Cellpose2D_model_auto(img_preprocess, 
                                                                                                                                           view='xz', 
                                                                                                                                           params=cellpose_segment_params, 
                                                                                                                                           basename=basename_xz, savefolder=saveFolderStep2)

    #### 3. running for yz orientation 
    img_segment_2D_yz_diams, img_segment_3D_yz_probs, img_segment_2D_yz_flows, img_segment_2D_yz_styles = uSegment3D.Cellpose2D_model_auto(img_preprocess, 
                                                                                                                                           view='yz', 
                                                                                                                                           params=cellpose_segment_params, 
                                                                                                                                           basename=basename_yz, savefolder=saveFolderStep2)
    print("Finish u-segment3D Package Step 2 Cellpose Segmentation successfully")    

    return []

##############################
##############################
##############################

segment3D_cellpose_segmentMD(input_path, output_path, params)
