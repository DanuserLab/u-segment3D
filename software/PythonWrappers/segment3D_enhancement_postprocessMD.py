"""
This is the key function + script, used in the wrapper fcn (segmentEnhancementPostprocessWrap) 
of step 5 SegmentationEnhancementPostprocessingProcess.m in u-segment3D MATLAB package.

This script is used to integrate python package/script with MovieData matlab structure.

I wrapped this script as a Python function, and defined two input arguments, input_path and output_path, with return [].
Then call this segment3D_enhancement_postprocessMD at bottom, since pyrunfile only runs script.
I added an input params = p, which was a structure in Matlab.

Example of how to use this test script in Matlab:
    inPath = '/path/to/folder/of/input'
    outPath = '/path/to/folder/for/saving/output' # will be created if not exist.
    pyrunfile("segment3D_enhancement_postprocessMD.py", input_path = inPath, output_path = outPath, ...
     label_diffusion_params = p1PassToPython, guided_filter_params = p2PassToPython, preprocess_params = preprocParamsPassToPython, outImgPathStep1 = outputImgPathStep1)


It is based on python script:
/applications/PythonPackIntoMovieData/Segment3D/tutorials/single_cells/segment3D_enhancement_postprocess.py

Alse see the full Python script: segment_blebs_3D_qz.py.
Also see script by Felix Zhou in 
/python-applications/Segment3D/tutorials/single_cells/segment_blebs_3D.py

Qiongjing (Jenny) Zou, June 2024

"""

"""    
Step 5 Segmentation enhancement Postprocessing 

"""
       
# only load the modules needed for this step - QZ:

import skimage.io as skio 
import pylab as plt 
import numpy as np 
import scipy.ndimage as ndimage
import skimage.segmentation as sksegmentation 
import os 

import segment3D.parameters as uSegment3D_params # this is useful to call default parameters, and keep track of parameter changes and for saving parameters.  
import segment3D.gpu as uSegment3D_gpu
import segment3D.usegment3d as uSegment3D
# import segment3D.filters as uSegment3D_filters 
import segment3D.file_io as uSegment3D_fio


def segment3D_enhancement_postprocessMD(input_path, output_path, label_diffusion_params, guided_filter_params, preprocess_params, outImgPathStep1):    
    #########################################
    # Set Preconditions
    #########################################
    
    inputFolder = input_path
    
    saveFolderStep5 = output_path
    if not os.path.exists(saveFolderStep5):
            os.makedirs(saveFolderStep5)
    
    
    print("Starting u-segment3D Package Step 5 Segmentation enhancement Postprocessing")
    
    ##############################
    ##############################
    ##############################
    
    # =============================================================================
    #     Load the saved output from previous steps 1 and 4 - QZ
    # =============================================================================
    # Load step 1's parameters     
    preprocess_params = preprocess_params # passed from Matlab - QZ
    # Convert string parameter from Matlab to a numpy function in Python:
    preprocess_params['avg_func_imgs'] = eval(preprocess_params['avg_func_imgs'])
    
    # load step 1's output image
    outputFileStep1 = outImgPathStep1 # passed from Matlab - QZ  
    img_preprocess = skio.imread(outputFileStep1)
    
    # Copied 2 lines from Step 2 Cellpose segmentation - Need this, otherwise line 92-94 will fail.
    # this expects a multichannel input image and in the format (M,N,L,channels) i.e. channels last.
    img_preprocess = img_preprocess[...,None] # we generate an artificial channel
    
    # load step 4's outputs: 
    segmentation3D_filt = skio.imread(os.path.join(inputFolder, 'uSegment3D_blebs_labels_postprocess_filtering_labels.tif'))
    
    # =============================================================================
    #     4. We can now do postprocessing - second postprocessing
    # =============================================================================
    """
    2. second postprocessing is to enhance the segmentation. 
        a) diffusing the labels to improve concordance with image boundaries 
        b) guided filter to recover detail such as subcellular protrusions on the surface of the cell. 
    """
    
    
    ###### a) label diffusion.
    # get parameters from movieData: - QZ
    label_diffusion_params = label_diffusion_params

    
    segmentation3D_filt_diffuse = uSegment3D.label_diffuse_3D_cell_segmentation_MP(segmentation3D_filt,
                                                                                   guide_image = img_preprocess[...,0],
                                                                                   params=label_diffusion_params)
    
    
    # =============================================================================
    #     Save Output - part 1:
    # =============================================================================
    
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
    plt.savefig(os.path.join(saveFolderStep5, 'Mid_Slices_Segmentation_Overlay__Post_label_diffusion.tif'))
    plt.show(block=False)

    
    # save this segmentation
    uSegment3D_fio.save_segmentation(os.path.join(saveFolderStep5,
                                                  'uSegment3D_blebs_labels_postprocess-diffuse.tif'), segmentation3D_filt_diffuse)
    
    
    
    ##### b) guided filter, (but we are going to do this at original resolution)
    # get parameters from movieData: - QZ
    guided_filter_params = guided_filter_params

    
    """
    resize segmentation and preprocessed image back to original sizes.
    """
    
    # for image we use linear interpolation i.e. order=1
    guide_image = uSegment3D_gpu.dask_cuda_rescale(img_preprocess[...,0],
                                                   zoom=[1./preprocess_params['factor'],1./preprocess_params['factor'],1./preprocess_params['factor']],
                                                   order=1,
                                                   mode='reflect',
                                                   chunksize=(512,512,512)) # note the inversion of the original factor
    
    # for segmentation we use 0th order i.e. order=0 nearest-neighbor interpolation to maintain integer-values
    segmentation3D_filt_diffuse = uSegment3D_gpu.dask_cuda_rescale(segmentation3D_filt_diffuse,
                                                                   zoom=[1./preprocess_params['factor'],1./preprocess_params['factor'],1./preprocess_params['factor']],
                                                                   order=0,
                                                                   mode='reflect',
                                                                   chunksize=(512,512,512)).astype(np.int32) # note the inversion of the original factor

    segmentation3D_filt_guide, guide_image_used = uSegment3D.guided_filter_3D_cell_segmentation_MP(segmentation3D_filt_diffuse,
                                                                                                #guide_image = img_preprocess[...,0],
                                                                                                guide_image = guide_image,
                                                                                                params=guided_filter_params)
   
    # =============================================================================
    #     Save Output - part 2:
    # =============================================================================
     
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
    plt.savefig(os.path.join(saveFolderStep5, 'Mid_Slices_Segmentation_Overlay__Post_guided_filter.tif'))    
    plt.show(block=False)
        
    # save this final segmentation
    uSegment3D_fio.save_segmentation(os.path.join(saveFolderStep5,
                                                  'uSegment3D_blebs_labels_postprocess-diffuse-guided_filter.tif'), segmentation3D_filt_guide)
    
    
    plt.close('all')

    print("Finish u-segment3D Package Step 5 Segmentation enhancement Postprocessing successfully")

    return []

##############################
##############################
##############################

segment3D_enhancement_postprocessMD(input_path, output_path, label_diffusion_params, guided_filter_params, preprocess_params, outImgPathStep1)