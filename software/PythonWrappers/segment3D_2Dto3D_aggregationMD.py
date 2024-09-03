"""
This is the key function + script, used in the wrapper fcn (twoDtoThreeDAggregateWrap) 
of step 3 TwoDtoThreeDAggregationProcess.m in u-segment3D MATLAB package.

This script is used to integrate python package/script with MovieData matlab structure.

I wrapped this script as a Python function, and defined two input arguments, input_path and output_path, with return [].
Then call this segment3D_2Dto3D_aggregationMD at bottom, since pyrunfile only runs script.
I added an input params = p, which was a structure in Matlab.

Example of how to use this test script in Matlab:
    inPath = '/path/to/folder/of/input'
    outPath = '/path/to/folder/for/saving/output' # will be created if not exist.
    pyrunfile("segment3D_2Dto3D_aggregationMD.py", input_path = inPath, output_path = outPath, params = p)


It is based on python script:
/applications/PythonPackIntoMovieData/Segment3D/tutorials/single_cells/segment3D_2Dto3D_aggregation.py

Alse see the full Python script: segment_blebs_3D_qz.py.
Also see script by Felix Zhou in 
/python-applications/Segment3D/tutorials/single_cells/segment_blebs_3D.py

Qiongjing (Jenny) Zou, June 2024

"""

"""
Step 3: 2D-to-3D aggregation

"""
       
# only load the modules needed for this step - QZ:

# import skimage.io as skio 
# import pylab as plt 
import numpy as np 
import scipy.ndimage as ndimage
# import skimage.segmentation as sksegmentation 
import os 

import segment3D.parameters as uSegment3D_params # this is useful to call default parameters, and keep track of parameter changes and for saving parameters.  
# import segment3D.gpu as uSegment3D_gpu
import segment3D.usegment3d as uSegment3D
# import segment3D.filters as uSegment3D_filters 
import segment3D.file_io as uSegment3D_fio


def segment3D_2Dto3D_aggregationMD(input_path, output_path, params):
    #########################################
    # Set Preconditions
    #########################################

    inputFolder = input_path

    saveFolderStep3 = output_path
    if not os.path.exists(saveFolderStep3):
            os.makedirs(saveFolderStep3)


    print("Starting u-segment3D Package Step 3 2D-to-3D aggregation")

    ##############################
    ##############################
    ##############################

    # =============================================================================
    #     Load the saved output from previous step - QZ
    # =============================================================================
    img_segment_3D_xy_probs = uSegment3D_fio.read_pickle(os.path.join(inputFolder, 'xy_cellpose_probs_xy.pkl'))['prob']

    img_segment_3D_xz_probs = uSegment3D_fio.read_pickle(os.path.join(inputFolder, 'xz_cellpose_probs_xz.pkl'))['prob']

    img_segment_3D_yz_probs = uSegment3D_fio.read_pickle(os.path.join(inputFolder, 'yz_cellpose_probs_yz.pkl'))['prob']


    img_segment_2D_xy_flows = uSegment3D_fio.read_pickle(os.path.join(inputFolder, 'xy_cellpose_flows_xy.pkl'))['flow']

    img_segment_2D_xz_flows = uSegment3D_fio.read_pickle(os.path.join(inputFolder, 'xz_cellpose_flows_xz.pkl'))['flow']

    img_segment_2D_yz_flows = uSegment3D_fio.read_pickle(os.path.join(inputFolder, 'yz_cellpose_flows_yz.pkl'))['flow']    

    # =============================================================================
    #     3. We can now use the predicted probabilies and flows directly to aggregate a 3D consensus segmentation (Direct Method)
    # =============================================================================

    # get parameters from movieData: - QZ
    aggregation_params = params

    print('========== Default 2D-to-3D aggregation parameters ========')
    print(aggregation_params)    
    print('============================================')


    # probs and gradients should be supplied in the order of [xy, xz, yz]. if one or more do not exist use [] e.g. [xy, [], []]
    segmentation3D, (probability3D, gradients3D) = uSegment3D.aggregate_2D_to_3D_segmentation_direct_method(probs=[img_segment_3D_xy_probs,
                                                                                                                   img_segment_3D_xz_probs,
                                                                                                                   img_segment_3D_yz_probs], 
                                                                                                            gradients=[img_segment_2D_xy_flows,
                                                                                                                       img_segment_2D_xz_flows,
                                                                                                                       img_segment_2D_yz_flows], 
                                                                                                            params=aggregation_params,
                                                                                                            savefolder=None,
                                                                                                            basename=None)

    # =============================================================================
    #     Save Output:
    # =============================================================================

    # we can save the segmentation with its colors using provided file I/O functions in u-Segment3D. If the savefolder exists and provided with basename in the function above, these would be saved automatically. 
    # 1. create a save folder 
    # savecellfolder = os.path.join('.', 
    #                               'blebs_segment');
    savecellfolder = saveFolderStep3;


    # 2. joint the save folder with the filename we wish to use. 2 files will be saved, 1=segmentation as labels and 2nd = 16color RGB colored segmentation for visualization
    uSegment3D_fio.save_segmentation(os.path.join(savecellfolder,
                                                  'uSegment3D_blebs_labels.tif'), segmentation3D)


    # 3. if you want to save the intermediate combined probability and gradients, we recommend using pickle 
    uSegment3D_fio.write_pickle(os.path.join(savecellfolder,
                                                  'uSegment3D_blebs_3Dcombined_probs_and_gradients'), 
                                savedict={'probs': probability3D.astype(np.float32),
                                          'gradients': gradients3D.astype(np.float32)})



    # ##### QZ also save aggregation_params as it is needed in step 4       
    # uSegment3D_fio.write_pickle(os.path.join(savecellfolder,'aggregation_params.pkl'), aggregation_params)


    print("Finish u-segment3D Package Step 3 2D-to-3D aggregation successfully")

    return []

##############################
##############################
##############################

segment3D_2Dto3D_aggregationMD(input_path, output_path, params)