function movieData = thresholdMovie(movieDataOrProcess,paramsIn)
%THRESHOLDMOVIE applies automatic or manual thresholding to every frame in input movie
%
% movieData = thresholdMovie(movieData)
% movieData = thresholdMovie(movieData,paramsIn)
%
% Applies manual or automatic thresholding to every frame of the input
% movie and then writes the resulting mask to file as a binary .tif in a
% sub-folder of the movie's analysis directory named "masks"
%
% Input:
% 
%   movieData - A MovieData object describing the movie to be processed, as
%   created by setupMovieDataGUI.m
%
%   paramsIn - Structure with inputs for optional parameters. The
%   parameters should be stored as fields in the structure, with the field
%   names and possible values as described below
% 
%   Possible Parameter Structure Field Names:
%       ('FieldName' -> possible values)
%
%       ('OutputDirectory' -> character string)
%       Optional. A character string specifying the directory to save the
%       masks to. Masks for different channels will be saved as
%       sub-directories of this directory.
%       If not input, the masks will be saved to the same directory as the
%       movieData, in a sub-directory called "masks"
%
%       ('ChannelIndex' -> Positive integer scalar or vector)
%       Optional. The integer index of the channel(s) to segment. If not
%       input, all channels will be segmented. If multiple
%       channels are selected, masks are generated for each independently.
%
%       ('ThresholdValue' -> Positive integer scalar or vector) Optional.
%       Intensity value to threshold images at. If not input, a value is
%       automatically calculated based on the image intensity histogram. If
%       a scalar, the same value is used for all channels. If a vector,
%       each element specifies the value to use for a specific channel.
%       Must be the same order and size as ChannelIndex. (requires good
%       SNR, significant area of background in image).
%
%       ('IsPercentile' -> True / False vector, same size as
%       ThresholdValue) Optional. Interpret ThresholdValue as a percentile
%       rather than an absolute intensity value. Default: False
% 
%       ('MaxJump' -> Positive scalar) If this is non-zero, any changes in
%       the auto-selected threshold value greater than this will be
%       suppressed by using the most recent good threshold. That is, if
%       MaxJump is set to 2.0 and the threshold changes by a factor of 2.2
%       between two consecutive frames, the new threshold will be ignored
%       and the previous threshold used. This option is ignored if the user
%       specifies a threshold.
%       Optional. Default is 0 (no jump suppression)
%
%       ('GaussFilterSigma' -> Positive scalar) If this is entered, the
%       image will be filtered first before thresholding and segmentation.
%       The filter kernel will be taken as a Gaussian with the input sigma.
%       Optional. Default is 0 (no filtering)
%           -- KJ
%
%       ('PreThreshold' -> True/False) Optional. If true, apply automatic
%       thresholding only to the values above a fixed threshold
%       Default: False
%
%       ('ExcludeOutliers' -> Nonnengative scalar) Optional. If this is
%       non-zero, then exclude outliers more than ExcludeOutliers*sigma
%       away from the mean see detectOutliers
%       Default: 0 (do not exclude outliers)
%
%       ('ExcludeZero' -> True/False) Optional. If true, then remove the
%       value zero from the data before automatic thresholding.
%       Default: False (do not exclude zero)
%       
% 
%       ('BatchMode' -> True/False)
%       If true, graphical output and user interaction is
%       supressed (i.e. progress bars, dialog and question boxes etc.)
%
%
% Output:
%
%   movieData - the updated MovieData object with the thresholding
%   parameters, paths etc. stored in it, in the field movieData.processes_.
%
%   The masks are written to the directory specified by the parameter
%   OuptuDirectory, with each channel in a separate sub-directory. They
%   will be stored as binary, bit-packed, .tif files. 
%
%
% Hunter Elliott, 11/2009
% Revamped 5/2010
%
%% ----- Parameters ----- %%
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

pString = 'mask_'; %Prefix for saving masks to file
pfName = 'threshold_values_for_channel_'; %Prefix for saving threshold values to file. Actual file name will have channel number appended.
dName = 'masks_for_channel_';%String for naming the mask directories for each channel

%% ----------- Input ----------- %%


%Check that input object is a valid moviedata TEMP
% if nargin < 1 || ~isa(movieData,'MovieData')
%     error('The first input argument must be a valid MovieData object!')
% end

if nargin < 2
    paramsIn = [];
end


%Get the indices of any previous threshold processes from this function                                                                              
% iProc = movieData.getProcessIndex('ThresholdProcess',1,0);
% 
% %If the process doesn't exist, create it
% if isempty(iProc)
%     iProc = numel(movieData.processes_)+1;
%     movieData.addProcess(ThresholdProcess(movieData,movieData.outputDirectory_));                                                                                                 
% end
% 
% thresProc= movieData.processes_{iProc};


%Parse input, store in parameter structure
% p = parseProcessParams(movieData.processes_{iProc},paramsIn);

[movieData,thresProc] = getOwnerAndProcess(movieDataOrProcess,'ThresholdProcess',true);
p = parseProcessParams(thresProc,paramsIn);

nChan = numel(movieData.channels_);

if max(p.ChannelIndex) > nChan || min(p.ChannelIndex)<1 || ~isequal(round(p.ChannelIndex),p.ChannelIndex)
    error('Invalid channel numbers specified! Check ChannelIndex input!!')
end


nChanThresh = length(p.ChannelIndex);
if ~isempty(p.ThresholdValue)
    if length(p.ThresholdValue) == 1
        p.ThresholdValue = repmat(p.ThresholdValue,[1 nChanThresh]);
    elseif length(p.ThresholdValue) ~= nChanThresh
        error('If you specify a threshold value, you must either specify one value to use for all channels, or 1 value per channel!');    
    end
end


%% --------------- Init ---------------%%

disp('Starting thresholding...')

% Set up the input directories (input images)
inFilePaths = cell(1,numel(movieData.channels_));
for i = p.ChannelIndex
    if isempty(p.ProcessIndex)
        inFilePaths{1,i} = movieData.getChannelPaths{i};
    else
       inFilePaths{1,i} = movieData.processes_{p.ProcessIndex}.outFilePaths_{1,i}; 
    end
end
thresProc.setInFilePaths(inFilePaths);

%Set up the mask directories as sub-directories of the output directory
for j = 1:nChanThresh;
    
    %Create string for current directory
    currDir = [p.OutputDirectory filesep dName num2str(p.ChannelIndex(j))];    
    %Save this in the process object
    thresProc.setOutMaskPath(p.ChannelIndex(j),currDir);
   
    %Check/create directory
    mkClrDir(currDir)               
end

allThresholds = cell(nChanThresh,1);

imageFileNames = movieData.getImageFileNames(p.ChannelIndex);


nImages = movieData.nFrames_;   
nImTot = nImages * nChanThresh;

%Get mask and image directories
maskDirs  = thresProc.outFilePaths_(p.ChannelIndex);
imDirs  = movieData.getChannelPaths(p.ChannelIndex);
    
threshMethod = thresProc.getMethods(p.MethodIndx).func;
%% ----- Thresholding ----- %%

if ~p.BatchMode && feature('ShowFigureWindows')
    wtBar = waitbar(0,['Please wait, thresholding channel ' num2str(p.ChannelIndex(1)) ' ...']);        
else
    wtBar = -1;
end        


for iChan = 1:nChanThresh
        
        
    if ishandle(wtBar)        
        waitbar((iChan-1)*nImages / nImTot,wtBar,['Please wait, thresholding channel ' num2str(p.ChannelIndex(iChan)) ' ...']);        
    end        
    disp(['Thresholding images for channel # ' num2str(p.ChannelIndex(iChan)) ' : '])
    disp(inFilePaths{1,p.ChannelIndex(iChan)})
    disp('Masks will be stored in directory :')
    disp(maskDirs{iChan})
    
    %Initialize vector for threshold values
    allThresholds{iChan} = nan(nImages,1);
    
    for iImage = 1:nImages
        

        %Load the current image
        if isempty(p.ProcessIndex)
            currImage = movieData.channels_(p.ChannelIndex(iChan)).loadImage(iImage);
        else
            currImage = movieData.processes_{p.ProcessIndex}.loadOutImage(p.ChannelIndex(iChan),iImage);
        end
        


        %KJ: filter image before thesholding if requested
        if p.GaussFilterSigma > 0
            currImage = filterGauss2D(double(currImage),p.GaussFilterSigma);
        end

        if ~isfield(p,'PreThreshold') || isempty(p.PreThreshold)
            % PreThreshold is false by default
            p.PreThreshold = false;
        end
        if ~isfield(p,'IsPercentile')
            p.IsPercentile = false(size(p.ThresholdValue));
        end
               
        if ~isfield(p,'ExcludeZero')
            p.ExcludeZero = false;
        end
        
        if ~isfield(p,'ExcludeOutliers')
            p.ExcludeOutliers = false;
        end
        
        if ~isfield(p,'Invert')
            p.Invert = false;
        end
        
        if ~isfield(p,'SingleFile')
            p.SingleFile = false;
        end
        
        if isempty(p.ThresholdValue) || p.PreThreshold
            % Automatic thresholding
            try
                data = currImage;
                
                %% Perform automatic thresholding
                if(p.PreThreshold)
                    % Use automatic thresholding method only on pixels
                    % above fixed threshold
                    absoluteThreshold = p.ThresholdValue(iChan);
                    if(p.IsPercentile(iChan))
                        absoluteThreshold = prctile(data(:),absoluteThreshold);
                    end
                    data = data(data > absoluteThreshold);
                end
                if(p.ExcludeZero)
                    data = data(data ~= 0);
                end
                if(p.ExcludeOutliers)
                    % Exclude outliers before perforing automatic
                    % thresholding
                    [~,inliers] = detectOutliers(data,p.ExcludeOutliers);
                    data = data(inliers);
                end
                
                currThresh = threshMethod(data);
                   
                if(p.PreThreshold)
                    % Ensure absolute threshold is above fixed threshold
                    currThresh = max(currThresh,absoluteThreshold);
                end
                if(p.ExcludeZero)
                    % Threshold must be at least zero if excluding zeros
                    currThresh = max(currThresh,0);
                end
            catch %#ok<CTCH>
                %If auto-threshold selection fails, and jump-correction is
                %enabled, force use of previous threshold
                if p.MaxJump > 0
                    currThresh = Inf;
                else
                    if ishandle(wtBar)
                        warndlg(['Could not automatically select a threshold in frame ' ...
                        num2str(iImage) '! Try specifying a threshold level, or enabling the MaxJump option!']);
                        close(wtBar)
                    end                                                
                    error(['Could not automatically select a threshold in frame ' ...
                        num2str(iImage) '! Try specifying a threshold level, or enabling the MaxJump option!']);
                    
                        
                end
            end
        else            
            currThresh = p.ThresholdValue(iChan);
            if(length(p.IsPercentile) >= iChan &&  p.IsPercentile(iChan))
                currThresh = prctile(currImage(:),currThresh);
            end
        end
        
        if p.MaxJump > 0
            %Check the threshold
            if iImage == 1
                allThresholds{iChan}(iImage) = currThresh; %Nothing to compare 1st frame to
            else
                if abs(currThresh / allThresholds{iChan}(find(~isnan(allThresholds{iChan}),1,'last'))-1) > (p.MaxJump-1)
                    %If the change was too large, don't store this threshold
                    %and use the most recent good value
                    allThresholds{iChan}(iImage) = NaN;
                    currThresh = allThresholds{iChan}(find(~isnan(allThresholds{iChan}),1,'last'));
                else
                    allThresholds{iChan}(iImage) = currThresh;
                end 
            end
        else
            allThresholds{iChan}(iImage) = currThresh;
        end
        
        %Apply the threshold to create the mask
        if(p.Invert)
            imageMask = currImage < currThresh;
        else
            % prior behavior
            imageMask = currImage > currThresh;
        end
    
        %write the mask to file
        if(p.SingleFile)
            imwrite(imageMask,[maskDirs{iChan} filesep pString imageFileNames{iChan}{1}],'WriteMode','append', 'Compression','none'); % fixed issue of ImageJ cannot open compressed mask. - Qiongjing (Jenny) Zou, Jan 2023
        else
            imwrite(imageMask,[maskDirs{iChan} filesep pString imageFileNames{iChan}{iImage}], 'Compression','none'); % fixed issue of ImageJ cannot open compressed mask. - Qiongjing (Jenny) Zou, Jan 2023
        end
        
        if ishandle(wtBar) && mod(iImage,5)
            %Update the waitbar occasionally to minimize slowdown
            waitbar((iImage + (iChan-1)*nImages) / nImTot,wtBar)
        end
                
    
    end    
   
    
end

if ishandle(wtBar)
    close(wtBar)
end


%% ------ Finish - Save parameters and movieData ----- %%


%Save the threshold values to the analysis directory as seperate files for
%each channel
for i = 1:nChanThresh    
    thresholdValues = allThresholds{i}; %#ok<NASGU>
    save([p.OutputDirectory filesep pfName num2str(p.ChannelIndex(i)) '.mat'],'thresholdValues');
end



thresProc.setDateTime;
movieData.save; %Save the new movieData to disk


disp('Finished thresholding!')
