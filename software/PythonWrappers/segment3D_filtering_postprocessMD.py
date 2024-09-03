"""
This is the key function + script, used in the wrapper fcn (segmentFilteringPostprocessWrap) 
of step 4 SegmentationFilteringPostprocessingProcess.m in u-segment3D MATLAB package.

This script is used to integrate python package/script with MovieData matlab structure.

I wrapped this script as a Python function, and defined two input arguments, input_path and output_path, with return [].
Then call this segment3D_filtering_postprocessMD at bottom, since pyrunfile only runs script.
I added an input params = p, which was a structure in Matlab.

Example of how to use this test script in Matlab:
    inPath = '/path/to/folder/of/input'
    outPath = '/path/to/folder/for/saving/output' # will be created if not exist.
    pyrunfile("segment3D_filtering_postprocessMD.py", input_path = inPath, output_path = outPath, params = p, ...
     aggregation_params = aggrParamsPassToPython, outImgPathStep1 = outputImgPathStep1)


It is based on python script:
/applications/PythonPackIntoMovieData/Segment3D/tutorials/single_cells/segment3D_filtering_postprocess.py

Alse see the full Python script: segment_blebs_3D_qz.py.
Also see script by Felix Zhou in 
/python-applications/Segment3D/tutorials/single_cells/segment_blebs_3D.py

Qiongjing (Jenny) Zou, June 2024

"""

    
"""
Step 4 Segmentation filtering Postprocessing 

"""
       
# only load the modules needed for this step - QZ:

import skimage.io as skio 
import pylab as plt 
import numpy as np 
import scipy.ndimage as ndimage
import skimage.segmentation as sksegmentation 
import os 

import segment3D.parameters as uSegment3D_params # this is useful to call default parameters, and keep track of parameter changes and for saving parameters.  
# import segment3D.gpu as uSegment3D_gpu
import segment3D.usegment3d as uSegment3D
# import segment3D.filters as uSegment3D_filters 
import segment3D.file_io as uSegment3D_fio
    
def segment3D_filtering_postprocessMD(input_path, output_path, params, aggregation_params, outImgPathStep1):
    #########################################
    # Set Preconditions
    #########################################
    
    inputFolder = input_path
    
    saveFolderStep4 = output_path
    if not os.path.exists(saveFolderStep4):
            os.makedirs(saveFolderStep4)
    
    
    print("Starting u-segment3D Package Step 4 Segmentation filtering Postprocessing")
    
    ##############################
    ##############################
    ##############################
    
    # =============================================================================
    #     Load the saved output from previous steps 1 and 3 - QZ
    # =============================================================================
    # load step 3's outputs: 
    segmentation3D = skio.imread(os.path.join(inputFolder, 'uSegment3D_blebs_labels_labels.tif'))
    aggregation_params = aggregation_params # passed from Matlab - QZ
    gradients3D = uSegment3D_fio.read_pickle(os.path.join(inputFolder, 'uSegment3D_blebs_3Dcombined_probs_and_gradients'))['gradients']
    
    # load step 1's output image   
    outputFileStep1 = outImgPathStep1 # passed from Matlab - QZ
    img_preprocess = skio.imread(outputFileStep1)
    
    # # Copied 2 lines from Step 2 Cellpose segmentation - Felix said we don't need this for this step:
    # # this expects a multichannel input image and in the format (M,N,L,channels) i.e. channels last.
    # img_preprocess = img_preprocess[...,None] # we generate an artificial channel
    
    # =============================================================================
    #     4. We can now do postprocessing - first postprocessing
    # =============================================================================
    """
    1. first postprocessing we can do is size filtering. 
    """
    # get parameters from movieData: - QZ
    postprocess_segment_params = params
    
    print('========== Default size and flow-consistency filtering parameters ========')
    print(postprocess_segment_params)    
    print('============================================')
    
    segmentation3D_filt, flow_consistency_intermediates = uSegment3D.postprocess_3D_cell_segmentation(segmentation3D,
                                                                                                     aggregation_params=aggregation_params,
                                                                                                     postprocess_params=postprocess_segment_params,
                                                                                                     cell_gradients=gradients3D,
                                                                                                     savefolder=None,
                                                                                                     basename=None)
                                              

    # =============================================================================
    #     Save Output:
    # =============================================================================
    
    # Save the segmentation3D_filt, 2 files will be saved, 1=segmentation as labels and 2nd = 16color RGB colored segmentation for visualization
    # use the first tif file as input for step 5
    uSegment3D_fio.save_segmentation(os.path.join(saveFolderStep4,
                                                  'uSegment3D_blebs_labels_postprocess_filtering.tif'), segmentation3D_filt)
    
    
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
    plt.savefig(os.path.join(saveFolderStep4, 'Mid_Slices_Segmentation_Overlay.tif'))
    plt.show(block=False)
    
    
    plt.close('all')

    print("Finish u-segment3D Package Step 4 Segmentation filtering Postprocessing successfully")

    return []

##############################
##############################
##############################

segment3D_filtering_postprocessMD(input_path, output_path, params, aggregation_params, outImgPathStep1)