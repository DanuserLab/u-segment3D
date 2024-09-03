function segmentEnhancementPostprocessWrap(movieDataOrProcess, varargin)
% segmentEnhancementPostprocessWrap wrapper function for SegmentationEnhancementPostprocessingProcess.
%
% INPUT
% movieDataOrProcess - either a MovieData (legacy)
%                      or a Process (new as of July 2016)
%
% param - (optional) A struct describing the parameters, overrides the
%                    parameters stored in the process (as of Aug 2016)
%
% OUTPUT
% none (saved to p.OutputDirectory)
%
% Changes
% As of July 2016, the first argument could also be a Process. Use
% getOwnerAndProcess to simplify compatability.
%
% As of August 2016, the standard second argument should be the parameter
% structure
%
% Qiongjing (Jenny) Zou, June 2024
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

%% ------------------ Input ---------------- %%
ip = inputParser;
ip.addRequired('MD', @(x) isa(x,'MovieData') || isa(x,'Process') && isa(x.getOwner(),'MovieData'));
ip.addOptional('paramsIn',[], @isstruct);
ip.parse(movieDataOrProcess, varargin{:});
paramsIn = ip.Results.paramsIn;

%% Registration
% Get MovieData object and Process
[movieData, thisProc] = getOwnerAndProcess(movieDataOrProcess, 'SegmentationEnhancementPostprocessingProcess', true);
p = parseProcessParams(thisProc, paramsIn); % If parameters are explicitly given, they should be used
% rather than the one stored in SegmentationEnhancementPostprocessingProcess

% Parameters:
% p

% Sanity Checks
nChan = numel(movieData.channels_);
if max(p.ChannelIndex) > nChan || min(p.ChannelIndex)<1 || ~isequal(round(p.ChannelIndex), p.ChannelIndex)
    error('Invalid channel numbers specified! Check ChannelIndex input!!')
end

% precondition / error checking
% check if ImagePreprocessingProcess was run
iImagePreprocessProc = movieData.getProcessIndex('ImagePreprocessingProcess',1,true); % nDesired = 1 ; askUser = true
if isempty(iImagePreprocessProc)
    error('ImagePreprocessingProcess needs to be done before run this process.')
end
% check if SegmentationFilteringPostprocessingProcess was run
if isempty(p.ProcessIndex)
    iSegFilterPostProc = movieData.getProcessIndex('SegmentationFilteringPostprocessingProcess',1,true); % nDesired = 1 ; askUser = true
    if isempty(iSegFilterPostProc)
        error('SegmentationFilteringPostprocessingProcess needs to be done before run this process.')
    end
elseif isa(movieData.processes_{p.ProcessIndex},'SegmentationFilteringPostprocessingProcess')
    iSegFilterPostProc = p.ProcessIndex;
else
    error('The process specified by ProcessIndex is not a valid SegmentationFilteringPostprocessingProcess! Check input!')
end

% logging input paths (bookkeeping)
inFilePaths = cell(1, numel(movieData.channels_));
for i = p.ChannelIndex
   inFilePaths{1,i} = movieData.processes_{iSegFilterPostProc}.outFilePaths_{1,i}; 
end
thisProc.setInFilePaths(inFilePaths);

% logging output paths.
mkClrDir(p.OutputDirectory);
outFilePaths = cell(1, numel(movieData.channels_));
for i = p.ChannelIndex
    outFilePaths{1,i} = [p.OutputDirectory filesep 'ch' num2str(i)];
    mkClrDir(outFilePaths{1,i});
end
thisProc.setOutFilePaths(outFilePaths);


%% Algorithm
% see segment3D_enhancement_postprocessMD.py
pyenv(ExecutionMode="OutOfProcess")

% When pass Matlab variables as inputs in pyrunfile, it automatically converts MATLAB data into types that best represent the data to the Python language.
% Such as logical in Matlab to bool in Python, double in Matlab to float in Python.
% see more https://www.mathworks.com/help/matlab/matlab_external/passing-data-to-python.html

for k = p.ChannelIndex
    inPath = inFilePaths{1,k};
    outPath = outFilePaths{1,k};

    p1PassToPython = rmfield(p,{'ChannelIndex','OutputDirectory','ProcessIndex', 'guide_filter', 'ridge_filter', 'guide_img2'}); % only pass the same parameters in label_diffusion_params to Python script
    p2PassToPython = rmfield(p,{'ChannelIndex','OutputDirectory','ProcessIndex', 'diffusion', 'guide_img'}); % only pass the same parameters in guided_filter_params to Python script
    % renamed guided_filter_params.guide_img2 to guide_img, to keep consistency with Python script:
    p2PassToPython.guide_img = p2PassToPython.guide_img2;
    p2PassToPython = rmfield(p2PassToPython,{'guide_img2'}); 

    % This step also used the parameters in the step 1 ImagePreprocessingProcess:
    preprocParamsPassToPython = rmfield(movieData.processes_{iImagePreprocessProc}.funParams_,{'ChannelIndex','OutputDirectory'}); % only pass the same parameters in preprocess_params to Python script

    % This step also used the output image from step 1 ImagePreprocessingProcess
    outputImgPathStep1 = [movieData.processes_{iImagePreprocessProc}.outFilePaths_{1,k} filesep 'preprocessed.tif'];

    pyrunfile("segment3D_enhancement_postprocessMD.py", input_path = inPath, output_path = outPath, ...
        label_diffusion_params = p1PassToPython, guided_filter_params = p2PassToPython, preprocess_params = preprocParamsPassToPython, outImgPathStep1 = outputImgPathStep1) % pyrunfile Since R2021b
end

%%%% end of algorithm

fprintf('\n Finished Segmentation Enhancement Postprocessing Process! \n')

end