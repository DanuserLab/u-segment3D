% MATLAB-based testing/performance suite for uSegment3DPackage
% Qiongjing (Jenny) Zou, June 2024
% Test uSegment3DPackage
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 

tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set Preconditions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_paths = path;
start_dir = pwd;


disp(['PWD:', pwd]);
% Dump path for debugging
s_path = strsplit(path,':');
s_match = (cellfun(@(x) regexp(x,'toolbox'), s_path, 'UniformOutput', false))';
matlab_paths = s_path(cellfun(@isempty, s_match))';
disp('    [MATLAB] current top-level paths....');
disp(matlab_paths);

disp(['Java heap max: ' num2str(java.lang.Runtime.getRuntime.maxMemory/1e9) 'GB'])
disp('Starting u-segment3D Package test script');

%----Initialization of temp dir
package_name = 'uSegment3DPackage';
t_stamp = datestr(now,'ddmmmyyyyHHMMSS');
tmpdir = fullfile(tempdir, [package_name '_test_' t_stamp]);
mkdir(tmpdir);

cd(tmpdir);

%-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gather Test image
%-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
local_test = true;
if local_test

	zipPath = '/work/bioinformatics/s184919/Analysis/Felix/uSegment3D/testData_singleCellBlebs/decon_1_1.zip';

else
	% Download test data for u-segment3D Package (Use BioHPC internal cloud account danuserweb)
    % Original Data can be found here.
    % /work/bioinformatics/s184919/Analysis/Felix/uSegment3D/example_data/single_cells/blebs/decon_1_1.tif
    url = ''; % make sure /download was put at the end! % QZ TODO
	zipPath = fullfile(tmpdir, 'decon_1_1.zip');
    
	urlwrite(url, zipPath);
end

unzip(zipPath, tmpdir);

% % Analysis Output Directory - do not need this if MD created based on BioFormats
% saveFolder = [tmpdir filesep 'analysis'];
% mkdir(saveFolder);

% Imaging/Microscope Parameters
% pixelSize = 100; % QZ artificial value
% timeInterval = 1; % QZ artificial value

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct Channels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MovieData creation

% Initialize from raw - Bioformats based
BFDataPath = [tmpdir filesep 'decon_1_1.tif'];
MD = MovieData(BFDataPath);
% A folder with the same name (i.e. decon_1_1) as the Tif file 
% was created, and the movieData was saved in that folder with 
% the same name as the Tif file (i.e. decon_1_1.mat).
% This folder is also MD.outputDirectory_
% MD.movieDataFileName_
% MD % see metadata
% MD.reader % see reader is BioFormatsReader

% Initialize from raw - Channel based approach
% ch1 = Channel([tmpdir filesep 'channel1']);
% MD = MovieData(ch1,saveFolder); % saveFolder here is MD's outputDirectory_
% MD.setPath(saveFolder); % set movieDataPath_, where to save .mat file
% MD.setFilename('movieData.mat');

% % Set some additional movie properties
% % MD.pixelSize_ = pixelSize;
% % MD.timeInterval_ = timeInterval;


MD.notes_= 'uSegment3DPackage test run!';

% Run sanityCheck on MovieData.
% Check image size and number of frames are consistent.
% Save the movie if successful
MD.sanityCheck;
MD.save;
MD.reset();

% Load the movie/dispaly contents
clear MD; % verify we can reload the object as intended.
MD = MovieData.load(fullfile(tmpdir,'decon_1_1','decon_1_1.mat'));


% Create InfoFlow Package and retrieve package index
Package_ = uSegment3DPackage(MD);
MD.addPackage(Package_);
stepNames = Package_.getProcessClassNames;
iPack =  MD.getPackageIndex('uSegment3DPackage');
disp('=====================================');
disp('|| Available Package Process Steps ||');
disp('=====================================');
disp(MD.getPackage(1).getProcessClassNames');

steps2Test = [1, 2, 3, 4, 5];
assert(length(Package_.processes_) >= length(steps2Test));
assert(length(Package_.processes_) >= max(steps2Test));
disp('Selected Package Process Steps');

for i=steps2Test
  disp(['Step ' num2str(i) ': ' stepNames{i}]);
end


%% Step 1: Image Preprocessing
disp('===================================================================');
disp('Running (1st) Image Preprocessing Process');
disp('===================================================================');
iPack = 1;
step_ = 1;
MD.getPackage(iPack).createDefaultProcess(step_)
params = MD.getPackage(iPack).getProcess(step_).funParams_;

% Customize parameters for segment_blebs_3D.py example:
% For this example which is deconvolved we found background illumination correction made it worse.
params.do_bg_correction = false;
% Downscale the image by factor 2, since at factor 1 we are getting mean diameter of around 60
params.factor = 0.5;

MD.getPackage(iPack).getProcess(step_).setPara(params);
MD.save;
params = MD.getPackage(iPack).getProcess(step_).funParams_
MD.getPackage(iPack).getProcess(step_).run();


%% Step 2: Cellpose Segmentation
disp('===================================================================');
disp('Running (2nd) Cellpose Segmentation Process');
disp('===================================================================');
iPack = 1;
step_ = 2;
MD.getPackage(iPack).createDefaultProcess(step_)
params = MD.getPackage(iPack).getProcess(step_).funParams_;

% Customize parameters for segment_blebs_3D.py example:
% Note: the default Cellpose model is set to \'cyto\' which is Cellpose 1., you can change this to any other available Cellpose model e.g. cyto2 is pretty good for single cells.
params.cellpose_modelname = 'cyto2';
% If the below auto-inferred diameter is picking up noise and being too small, we can increase the default ksize, alternatively we can add median_filter.
params.ksize = 21;
% to suppress visualization set debug_via to false.
params.debug_viz = false; % if this is true, more files saved, run time is not affacted much.

MD.getPackage(iPack).getProcess(step_).setPara(params);
MD.save;
params = MD.getPackage(iPack).getProcess(step_).funParams_
MD.getPackage(iPack).getProcess(step_).run();


%% Step 3: 2D to 3D Aggregation
disp('===================================================================');
disp('Running (3rd) 2D to 3D Aggregation Process');
disp('===================================================================');
iPack = 1;
step_ = 3;
MD.getPackage(iPack).createDefaultProcess(step_)
params = MD.getPackage(iPack).getProcess(step_).funParams_;

% Customize parameters for segment_blebs_3D.py example:
% make sure that we say we are using Cellpose probability predictions which needs to be normalized. If not using Cellpose predicted masks, then set this to be False. We assume this has been appropriately normalized to 0-1
params.combine_cell_probs.cellpose_prob_mask = true; % default is also true. hide on GUI.
% add some temporal decay 
params.gradient_descent.gradient_decay = 0.1;

MD.getPackage(iPack).getProcess(step_).setPara(params);
MD.save;
params = MD.getPackage(iPack).getProcess(step_).funParams_
MD.getPackage(iPack).getProcess(step_).run();


%% Step 4: Segmentation Filtering Postprocessing
disp('===================================================================');
disp('Running (4th) Segmentation Filtering Postprocessing Process');
disp('===================================================================');
iPack = 1;
step_ = 4;
MD.getPackage(iPack).createDefaultProcess(step_)
params = MD.getPackage(iPack).getProcess(step_).funParams_;

% Customize parameters for segment_blebs_3D.py example:
% use default

MD.getPackage(iPack).getProcess(step_).setPara(params);
MD.save;
params = MD.getPackage(iPack).getProcess(step_).funParams_
MD.getPackage(iPack).getProcess(step_).run();


%% Step 5: Segmentation Enhancement Postprocessing
disp('===================================================================');
disp('Running (5th) Segmentation Enhancement Postprocessing Process');
disp('===================================================================');
iPack = 1;
step_ = 5;
MD.getPackage(iPack).createDefaultProcess(step_)
params = MD.getPackage(iPack).getProcess(step_).funParams_;


% Customize parameters for segment_blebs_3D.py example:

params.diffusion.refine_iters = int16(15); % use lower amount of iterations, since base segmentation is quite close. 
params.diffusion.refine_alpha = 0.75; % bias towards image-driven 

%  for guided image we will use the input to Cellpose and specify we want to further do ridge filter enhancement within the function
params.ridge_filter.do_ridge_enhance = true;
params.ridge_filter.mix_ratio = 0.5; % combine 25% of the ridge enhanced image with the input guide image. % default is 0.5
params.ridge_filter.sigmas = {1.5};

% we can adjust the radius to try to capture more detail 
params.guide_filter.radius = int16(45); % drop lower (this should be approx size of protrusions) 
params.guide_filter.eps = 1e-4; % we can increase this to reduce the guided filter strength or decrease to increase.. % default is 1e-4
params.guide_filter.mode = 'normal'; % default is 'normal' % if the guide image inside is hollow (i.e. black pixels of low intensity like here), you may need to run 'additive' mode and tune 'base_erode' when using large radius
params.guide_filter.base_erode = int16(5); % default is 5


MD.getPackage(iPack).getProcess(step_).setPara(params);
MD.save;
params = MD.getPackage(iPack).getProcess(step_).funParams_
MD.getPackage(iPack).getProcess(step_).run();


    
disp('Finish u-segment3D Package test run Matlab script successfully');
%- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% The movieViewer will not work for uSegment3DPackage???, since movieViewer
% cannot display RGB images as one channel, need to separate into 3 channels.

% %% Package GUI and movieViwer with Overlays
% disp('===================================================================');
% disp('Running (GUI Output Display Generation) u-segment3D Package ');
% disp('===================================================================');
% 
% ff = findall(0,'Type', 'Figure'); 
% delete(ff);
% 
% h = MD.getPackage(1).GUI(MD);
% % pb = findall(h,'-regexp', 'tag', 'pushbutton_show');
% pb = findall(h,'-regexp', 'tag', 'pushbutton_show', 'Enable','on');
% 
% % move mouse over the Results button
% for i = 1:numel(pb)
%     
%     % Load Results &  activate movieViewer
%     pb(i).Value = 1;
%     pb(i).Callback(pb(i),[]);
%    
%     pause(.25); % make sure graphics are ready
% 
%     % Run the movie 
%     hButton = findobj('String','Run movie');
%     hButton.Value = 1;
%     hButton.Callback(hButton,[]);   
% 
%     pause(.25);
%     delete(findobj(0,'Name', 'Viewer'));
% end
% 
% % Use delete to avoid closing "save MD" pop-up menu
% ff = findall(0,'Type', 'Figure'); 
% delete(ff);
toc
%% Clean up tmpdir
disp('*****************************************');
disp('%% !!!!!!!Cleaning up /tmp/ ');
disp('*****************************************');

cd('~')
ls(tmpdir)
rmdir(tmpdir,'s')
assert(~(exist(tmpdir, 'dir') == 7))

disp('*****************************************');
disp('%% !!!!!!!done cleaning up /tmp/ ');
disp('*****************************************');
