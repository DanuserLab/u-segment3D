"""
This is the key function + script, used in the wrapper fcn (imagePreprocessWrap) 
of step 1 ImagePreprocessingProcess.m in u-segment3D MATLAB package.

This script is used to integrate python package/script with MovieData matlab structure.

I wrapped this script as a Python function, and defined two input arguments, input_path and output_path, with return [].
Then call this segment3D_preprocessMD at bottom, since pyrunfile only runs script.
I added an input params = p, which was a structure in Matlab.

Example of how to use this test script in Matlab:
    inPath = '/path/to/folder/of/input/images'
    outPath = '/path/to/folder/for/saving/output' # will be created if not exist.
    pyrunfile("segment3D_preprocessMD.py", input_path = inPath, output_path = outPath, params = p)


It is based on python script:
/applications/PythonPackIntoMovieData/Segment3D/tutorials/single_cells/segment3D_preprocess.py

Alse see the full Python script: segment_blebs_3D_qz.py.
Also see script by Felix Zhou in 
/python-applications/Segment3D/tutorials/single_cells/segment_blebs_3D.py

Qiongjing (Jenny) Zou, June 2024

"""

"""
Step 1: preprocessing

"""
   
# only load the modules needed for this step - QZ:
    
import skimage.io as skio 
import pylab as plt 
import numpy as np 
import scipy.ndimage as ndimage
# import skimage.segmentation as sksegmentation 
import os 

import segment3D.parameters as uSegment3D_params # this is useful to call default parameters, and keep track of parameter changes and for saving parameters.  
# import segment3D.gpu as uSegment3D_gpu
import segment3D.usegment3d as uSegment3D
# import segment3D.filters as uSegment3D_filters 
import segment3D.file_io as uSegment3D_fio

def segment3D_preprocessMD(input_path, output_path, params):
    #########################################
    # Set Preconditions
    #########################################
    inputFile = input_path

    saveFolderStep1 = output_path
    if not os.path.exists(saveFolderStep1):
            os.makedirs(saveFolderStep1)


    print("Starting u-segment3D Package Step 1 Preprocessing")

    ##############################
    ##############################
    ##############################

    # =============================================================================
    #     Read the raw image data
    # =============================================================================


    img = skio.imread(inputFile)


    """
    Visualize the image in max projection and mid-slice to get an idea of how it looks
    """
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Max. Projection')
    plt.imshow(img.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(img.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(img.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(saveFolderStep1, 'Max_Projection_original.tif'))
    plt.show(block=False)


    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Mid Slices')
    plt.imshow(img[img.shape[0]//2], cmap='gray')
    plt.subplot(132)
    plt.imshow(img[:,img.shape[1]//2], cmap='gray')
    plt.subplot(133)
    plt.imshow(img[:,:,img.shape[2]//2], cmap='gray')
    plt.savefig(os.path.join(saveFolderStep1, 'Mid_Slices_original.tif'))
    plt.show(block=False)

    # =============================================================================
    #     1. Preprocess the image. 
    # =============================================================================

    # get parameters from movieData: - QZ
    preprocess_params = params

    # Convert string parameter from Matlab to a numpy function in Python:
    preprocess_params['avg_func_imgs'] = eval(preprocess_params['avg_func_imgs'])



    # this image is isotropic, therefore we don't change any of the parameters - comment for segment_blebs_3D.py
    print('========== preprocessing parameters ========')
    print(preprocess_params)    
    print('============================================')


    """
    There is some salt and pepper noise in the background, we can run a small median filter to remove, before preprocessing. 
    """
    # run the default preprocessing process in uSegment3D. This process is adapted to multichannel images. therefore for single-channel we need to squeeze the output
    img_preprocess = uSegment3D.preprocess_imgs(img, params=preprocess_params)
    img_preprocess = np.squeeze(img_preprocess)

    # =============================================================================
    #     Save Output:
    # =============================================================================

    # Save the preprocessed image as a .tif file - QZ
    preprocessed_img_path = os.path.join(saveFolderStep1, 'preprocessed.tif')
    skio.imsave(preprocessed_img_path, img_preprocess.astype(np.float32))

    # ##### QZ also save preprocess_params as it is needed in step 5
    # uSegment3D_fio.write_pickle(os.path.join(saveFolderStep1,'preprocess_params.pkl'), preprocess_params)

    # have a look at the processed. The result should have better contrast and more uniform illumination of the shape. 
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Max. Projection of preprocessed')
    plt.imshow(img_preprocess.max(axis=0), cmap='gray')
    plt.subplot(132)
    plt.imshow(img_preprocess.max(axis=1), cmap='gray')
    plt.subplot(133)
    plt.imshow(img_preprocess.max(axis=2), cmap='gray')
    plt.savefig(os.path.join(saveFolderStep1, 'Max_Projection_of_preprocessed.tif'))
    plt.show(block=False)


    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.title('Mid Slices of preprocessed')
    plt.imshow(img_preprocess[img_preprocess.shape[0]//2], cmap='gray')
    plt.subplot(132)
    plt.imshow(img_preprocess[:,img_preprocess.shape[1]//2], cmap='gray')
    plt.subplot(133)
    plt.imshow(img_preprocess[:,:,img_preprocess.shape[2]//2], cmap='gray')
    plt.savefig(os.path.join(saveFolderStep1, 'Mid_Slices_of_preprocessed.tif'))
    plt.show(block=False)

    plt.close('all')


    print("Finish u-segment3D Package Step 1 Preprocessing successfully")

    return []

##############################
##############################
##############################

segment3D_preprocessMD(input_path, output_path, params)

