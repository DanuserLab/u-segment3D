function multiScaleAutoSeg_multiObject(movieDataOrProcess, varargin)
% multiScaleAutoSeg_multiObject wrapper function for MultiScaleAutoSegmentationProcess since Aug 2021.
% Previously MultiScaleAutoSegmentationProcess's wrapper fcn was multiScaleAutoSeg.
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
% Qiongjing (Jenny) Zou, Aug 2021
%
% Modified to give user option to turn 'on' and 'off' of the pop up figures during run.
% Qiongjing (Jenny) Zou, Nov 2024
%
% Copyright (C) 2025, Danuser Lab - UTSouthwestern 
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
ip.addRequired('movieDataOrProcess', @isProcessOrMovieData);
ip.addOptional('paramsIn',[], @isstruct);
ip.parse(movieDataOrProcess, varargin{:});
paramsIn = ip.Results.paramsIn;

%% Registration
% Get MovieData object and Process
[movieData, thisProc] = getOwnerAndProcess(movieDataOrProcess,'MultiScaleAutoSegmentationProcess', true);
p = parseProcessParams(thisProc, paramsIn); % If parameters are explicitly given, they should be used
% rather than the one stored in MultiScaleAutoSegmentationProcess

% Parameters:
currTightness = p.tightness;
currObjectNumber = p.ObjectNumber;
currFinalRefinementRadius = p.finalRefinementRadius;
currUseSummationChannel = p.useSummationChannel;
currProcessIndex = p.ProcessIndex; % the index of using which previous proc's output as input of thisProc
currFigVisible = p.figVisible;
currVerbose = p.verbose;
% below params not on GUI:
currImagesOut = p.imagesOut;
currMinimumSize = p.MinimumSize;

% Sanity Checks
nChan = numel(movieData.channels_);
if max(p.ChannelIndex) > nChan || min(p.ChannelIndex)<1 || ~isequal(round(p.ChannelIndex), p.ChannelIndex)
    error('Invalid channel numbers specified! Check ChannelIndex input!!')
end

% Check whether the channel(s) selected for this proc included in the
% channel(s) processed from previous step
% Added by Qiongjing (Jenny) Zou, Oct 2024
if ~isempty(p.ProcessIndex)
    if ~all(ismember(p.ChannelIndex, movieData.processes_{p.ProcessIndex}.funParams_.ChannelIndex))
        error('The channels selected is not included in the channels processed from the previous step %s! Check input!', class(movieData.processes_{p.ProcessIndex}));
    end
end

% precondition / error checking
if isequal(currUseSummationChannel, 1)
    if isempty(currProcessIndex)
        currProcessIndex = movieData.getProcessIndex('GenerateSummationChannelProcess',1,true); % nDesired = 1 ; askUser = true
    elseif ~isa(movieData.processes_{currProcessIndex},'GenerateSummationChannelProcess')
        error('The process specified by ProcessIndex is not a valid GenerateSummationChannelProcess! Check input!')
    end
end

% logging input paths (bookkeeping)
% Set up the input directories (input images)
inFilePaths = cell(1, numel(movieData.channels_));
for i = p.ChannelIndex
    if isempty(currProcessIndex)
        inFilePaths{1,i} = movieData.getChannelPaths{i};
    else
        inFilePaths{1,i} = movieData.processes_{currProcessIndex}.outFilePaths_{1,i};
    end
end
thisProc.setInFilePaths(inFilePaths);


% logging output paths.
% only masksOutDir are set in outFilePaths_, other output are saved but not logged here.
dName = 'MSASeg_masks_for_channel_';%String for naming the mask directories for each channel
outFilePaths = cell(1, numel(movieData.channels_));

% (by Noh) To prevent redundant run of the MSA algorithm, params and scoreArray
% files are kept in the OutputDirectory. 
% Only need to keep p.OutputDirtory not the subdirectories.
% mkClrDir(p.OutputDirectory);  % Change to not-removing existing output
if ~isfolder(p.OutputDirectory); mkdir(p.OutputDirectory); end

for iChan = p.ChannelIndex
    % Create string for current directory
    currDir = [p.OutputDirectory filesep dName num2str(iChan)];
    outFilePaths{1,iChan} = currDir;
    thisProc.setOutMaskPath(iChan, currDir);
    mkClrDir(outFilePaths{1,iChan});
end
thisProc.setOutFilePaths(outFilePaths);


%% Algorithm
% see MSA_Seg_multiObject_imDir.m
% Edit to make it work for all MD.Reader, such as BioFormatsReader. Before, the algorithm only works for TiffSeriesReader.
% Edit again to make it also work when input is from output of a previous process. - Qiongjing (Jenny) Zou, Nov 2022

tic

for k = p.ChannelIndex
    masksOutDir = outFilePaths{1,k};
    
    if isempty(currProcessIndex)
        imFileNamesF = movieData.getImageFileNames(k);
        imFileNames = imFileNamesF{1};
    else
        fileReads = dir(inFilePaths{1,k});
        ind = arrayfun(@(x) (x.isdir == 0), fileReads);
        imFileNames = {fileReads(ind).name}';
    end
    
    I = cell(movieData.nFrames_, 1);
    imgStack = [];
    for fr = 1: movieData.nFrames_
        if isempty(currProcessIndex)
            I{fr} = movieData.channels_(k).loadImage(fr); % this is the way to read image for all MD.Reader, when input is raw images.
        else
            I{fr} = imread([inFilePaths{1,k} filesep imFileNames{fr}]); % this is the way to read image from output of a previous process.
        end
        imgStack = cat(3, imgStack, I{fr});
        if isequal(currVerbose, 'on')
        fprintf(1, '%g ', fr);
        end
    end

    MSA_Seg_multiObject_imDir_2(I, imgStack, imFileNames, movieData.nFrames_, p.OutputDirectory, ...
        masksOutDir, k, 'tightness', currTightness, 'numVotes', p.numVotes, ...
        'ObjectNumber', currObjectNumber, 'finalRefinementRadius', currFinalRefinementRadius, ...
        'imagesOut', currImagesOut, 'figVisible', currFigVisible, 'MinimumSize', currMinimumSize, 'verbose', currVerbose)
    
end


toc
%%%% end of algorithm

disp('Multi-Scale Automatic Segmentation is done!')

end % end of wrapper fcn


function MSA_Seg_multiObject_imDir_2(I, imgStack, fileNames, frmax, outputDir, masksOutDir, iChan, varargin)
% local function modified from MSA_Seg_multiObject_imDir with additional input arguments, masksOutDir and iChan
% Also, deleted input argument, inputImgDir, and added I, imgStack, fileNames, frmax.
%
% Updates:
% 2024/11. J Noh. 
%   Revise for efficiency. The algorithm now saves the voting
%   score array output, so that if it is run again only with a different
%   threshold, it now doesn't run multi-scale segmentations redundantly but
%   it generates new masks for the different threshold directly from the
%   previously saved voting score array.

%% Parse input

ip = inputParser;
ip.addParameter('tightness', 0.5, @(x) isnumeric(x) && (x==-1 || x >= 0 || x<=1));
ip.addParameter('numVotes', -1);
ip.addParameter('imagesOut', 1);
ip.addParameter('figVisible', 'on');
ip.addParameter('finalRefinementRadius', 1);
ip.addParameter('MinimumSize', 10);
ip.addParameter('ObjectNumber', 1000);
ip.addParameter('verbose', 'off');
%ip.addParameter('parpoolNum', 1);

ip.parse(varargin{:});
p = ip.Results;

if (p.numVotes > 0); p.tightness = -1; end

%% control parameter structs to prevent redundant running when only a threshold is changed

p1 = p;     % MSA seg parameter struct

% Select MSA seg parameters except the threshold parameters
p2 = struct();
p2.finalRefinementRadius = p.finalRefinementRadius;
p2.MinimumSize = p.MinimumSize;
p2.ObjectNumber = p.ObjectNumber;

% Get old parameter
if isfile([outputDir filesep 'p2_for_channel_' num2str(iChan) '.mat'] )
    tmp = load([outputDir filesep 'p2_for_channel_' num2str(iChan) '.mat']);
    old_p2 = tmp.p2;
else
    % If it is the 1st run (w/o 'p2.mat' output), then make a fake 'old_p2'
    % struct, which always differs from 'p2' to run the segmentation in the below.
    old_p2 = struct();
    old_p2.finalRefinementRadius = -1;       % a fake value
end

if ~isfolder(outputDir); mkdir(outputDir); end
save([outputDir filesep 'p1_for_channel_' num2str(iChan) '.mat'], 'p1');
save([outputDir filesep 'p2_for_channel_' num2str(iChan) '.mat'], 'p2');

%% -------- Parameters ---------- %%

if ~isdir(masksOutDir); mkdir(masksOutDir); end

pString = 'MSA_mask_';      %Prefix for saving masks to file

% Comment out, below method does not work for BioFormats reader, used im = MD.channels_(chIdx).loadImage(frameIdx); see above.
% Qiongjing (Jenny) Zou, Aug 2021
%% Load images

% fileReads = dir(inputImgDir);
% ind = arrayfun(@(x) (x.isdir == 0), fileReads);

% fileNames = {fileReads(ind).name}; % TODO 
% frmax = numel(fileNames);

% I = cell(frmax, 1);
% imgStack = [];
% for fr = 1:frmax
%     I{fr} = imread(fullfile(inputImgDir, fileNames{fr}));
%     imgStack = cat(3, imgStack, I{fr});
%     fprintf(1, '%g ', fr);
% end

%% Run MSA seg only if it has not run with the same parameter except threshodling parms
% Check if MSA Seg is once run for this movieData.
% Only if not, run MSA algorithm to compute voting score Array (step 1) and
% masks (step 2).
% If previous results for the same parameters (except thresholds) exist, 
% then compute only masks (step 2).

scoreArrayFilePath = [outputDir filesep 'scoreArray_for_channel_' num2str(iChan) '.mat'];
if ~isfile(scoreArrayFilePath) || ~isequaln(p2, old_p2)
    
    [refinedMask, voteScoreImgs] = MSA_Seg_1stRun(p, outputDir, frmax, imgStack, iChan, p.verbose);
    
    % voteScoreImg
    % dir name for vote score images
    imOutDir2 = [outputDir filesep 'MSASeg_voteScoreImgs_for_channel_' num2str(iChan)];
    if ~isfolder(imOutDir2); mkdir(imOutDir2); end   

    for fr = 1:frmax
        imwrite(voteScoreImgs{fr}, fullfile(imOutDir2, ['voteScores_', fileNames{fr}]) );
    end
    
else
    
    refinedMask = MSA_Seg_2ndRun(p, outputDir, iChan, p.verbose);

end   

%% save mask images

for fr = 1:frmax
    %Write the refined mask to file
    imwrite(mat2gray(refinedMask{fr}), fullfile(masksOutDir, [pString, fileNames{fr}]) );
end

%% imagesOut

if p.imagesOut == 1
    
    if p.numVotes >= 0
        prefname = ['numVotes_', num2str(p.numVotes)];
    elseif p.tightness >= 0
        prefname = ['tightness_', num2str(p.tightness)];
    else
        prefname = '_';
    end
    
    dName2 = ['MSASeg_maskedImages_' prefname '_for_channel_' num2str(iChan)];
    imOutDir = fullfile(outputDir, dName2);
    if ~isdir(imOutDir); mkdir(imOutDir); end
    
    allint = imgStack(:);
    intmin = quantile(allint, 0.002);
    intmax = quantile(allint, 0.998);
    
    % Edited the code below to enable the on and off of the segmentation figures. - Qiongjing (Jenny) Zou, Nov 2024
    ftmp = figure('Visible', p.figVisible);
    ax = axes('Parent', ftmp);  % Create an axes inside the figure. Useful when figure is invisible. - QZ
    for fr = 1:frmax
        % figure(ftmp)
        % imshow(I{fr}, [intmin, intmax])
        imshow(I{fr}, [intmin, intmax], 'Parent', ax)  % Display image in specified axes. Useful when figure if invisible. - QZ
        hold on
        bdd = bwboundaries(refinedMask{fr});
        
        for k = 1:numel(bdd)
            bdd1 = bdd{k};
            % plot(bdd1(:,2), bdd1(:,1), 'r');
            plot(ax, bdd1(:,2), bdd1(:,1), 'r');  % - QZ
        end
        
        %bdd1 = bdd{1};
        %plot(bdd1(:,2), bdd1(:,1), 'r');
        hold off
        
        % h = getframe(gcf);
        h = getframe(ftmp);  % Capture the frame from the figure. QZ
        imwrite(h.cdata, fullfile(imOutDir, fileNames{fr}), 'tif')
    end    
end

end


%% When MSA seg is run for the first time with the same segmentation parameter

function [refinedMask, voteScoreImgs] = MSA_Seg_1stRun(p, outputDir, frmax, imgStack, iChan, verbose)
    %% Time series of 5 numbers

    pixelmat = reshape(imgStack, [], frmax);
    pixelmat1 = pixelmat;
    pixelmat1(pixelmat1 == 0) = NaN;
    %sum(isnan(pixelmat1(:)))

    mts = mean(pixelmat1, 1, 'omitnan');
    medts = median(pixelmat1, 1, 'omitnan');
    q1ts = quantile(pixelmat1, 0.25, 1);
    q3ts = quantile(pixelmat1, 0.75, 1);
    q99ts = quantile(pixelmat1, 0.99, 1);
    q01ts = quantile(pixelmat1, 0.01, 1);

    fts = figure('Visible', p.figVisible); % allowed pop up figure to be turned off. - Qiongjing (Jenny) Zou, Nov 2024
    plot(mts)
    hold on

    plot(medts)
    plot(q1ts)
    plot(q3ts)
    plot(q01ts)
    plot(q99ts)
    hold off

    legend('Mean', 'Median', 'Perct25', 'Perct75', 'Perct01', 'Perct99')
    title('Time series of 5 summary statistics')

    %% saveas 
    saveas(fts, [outputDir filesep 'TS_of_5statistics_for_channel_' num2str(iChan) '.png'], 'png')
    
    %% Multi Scale Segmentation

    refinedMask = cell(frmax, 1);
    voteScoreImgs = cell(frmax, 1); 
    currTightness = p.tightness;
    currNumVotes = p.numVotes;
    
    scoreArray = zeros(size(imgStack));    

    parfor fr = 1:frmax
        if isequal(verbose, 'on')
        disp('=====')
        disp(['Frame: ', num2str(fr)])
        end   
        im = imgStack(:,:,fr);
        [refinedMask{fr}, voteScoreImgs{fr}, scoreArray(:,:,fr)] = ...
            multiscaleSeg_multiObject_im(im, ...
                'tightness', currTightness, 'numVotes', currNumVotes, ...
                'finalRefinementRadius', p.finalRefinementRadius, ...
                'MinimumSize', p.MinimumSize, 'ObjectNumber', p.ObjectNumber, 'verbose', verbose);
    end
    
    %% save voting scoreArray
    %save(fullfile(outputDir, 'scoreArray.mat'), 'scoreArray');

    save([outputDir filesep 'scoreArray_for_channel_' num2str(iChan) '.mat'], 'scoreArray')

end


%% When MSA seg is already run with the same parameters except thresholds

function refinedMask = MSA_Seg_2ndRun(p, outputDir, iChan, verbose)
    
    % Load scoreArray    
    tmp = load([outputDir filesep 'scoreArray_for_channel_' num2str(iChan) '.mat']);
    scoreArray = tmp.scoreArray;

    refinedMask = multiscaleSeg_multiObject_2ndRun(scoreArray, ...
            'numVotes', p.numVotes, 'tightness', p.tightness, ...
            'finalRefinementRadius', p.finalRefinementRadius, ...
            'MinimumSize', p.MinimumSize, 'ObjectNumber', p.ObjectNumber, 'verbose', verbose);

end
